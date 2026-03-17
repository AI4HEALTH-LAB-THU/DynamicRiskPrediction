import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Tuple

# ==========================
# 全局配置
# ==========================

# 根目录
BASE_DIR = "result/bench_dynamic/xgb/prediction_results/251209_final"
CV_DIR = os.path.join(BASE_DIR, "outer_cv_predictions")
TEST_DIR = os.path.join(BASE_DIR, "external_val_predictions")

# 输出
SUMMARY_CSV = os.path.join(BASE_DIR, "summary_metrics.csv")

# bootstrap 次数
N_BOOTSTRAP = 1000
RANDOM_STATE = 42

# 疾病英文名
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
OUTCOMES = list(outcome_eng_mapping.values())

# 5 个模型的命名（用于文件拼接 & 表里 model_type）
MODEL_TAGS = {
    "model1_base": "model1_base",
    "model2_clinical": "model2_clinical",
    "model3_dynamic": "model3_dynamic",
    "model4_clinical_baseline_exam": "model4_clinical_baseline_exam",
    "model5_clinical_dynamic_exam": "model5_clinical_dynamic_exam",
}
MODELS_ORDER = list(MODEL_TAGS.keys())  # 保持顺序：1,2,3,4,5

# summary 表的列顺序
# 说明：
# - cv_vs_model5_auroc_pvalue / cv_vs_model5_auprc_pvalue: 本行模型 vs model5 的 p 值（model5 行为 NaN）
# - NRI/IDI 只针对 model5 vs model4，但会给出事件组 / 非事件组 / 总 NRI 的 95% CI
SUMMARY_COLUMNS = [
    "outcome", "model_type",
    "cv_roc_auc_mean", "cv_roc_auc_ci_lower", "cv_roc_auc_ci_upper",
    "cv_pr_auc_mean", "cv_pr_auc_ci_lower", "cv_pr_auc_ci_upper",
    "cv_brier_mean", "cv_brier_ci_lower", "cv_brier_ci_upper",

    # 各模型 vs model5 的显著性（本行模型 vs model5）
    "cv_vs_model5_auroc_pvalue", "cv_vs_model5_auprc_pvalue",

    # 仅 model5 vs model4 的 NRI/IDI（总 + 事件组 + 非事件组）
    "cv_model5_vs_model4_nri", "cv_model5_vs_model4_nri_lower", "cv_model5_vs_model4_nri_upper",
    "cv_model5_vs_model4_nri_event", "cv_model5_vs_model4_nri_event_lower", "cv_model5_vs_model4_nri_event_upper",
    "cv_model5_vs_model4_nri_nonevent", "cv_model5_vs_model4_nri_nonevent_lower", "cv_model5_vs_model4_nri_nonevent_upper",
    "cv_model5_vs_model4_idi", "cv_model5_vs_model4_idi_lower", "cv_model5_vs_model4_idi_upper",

    "test_roc_auc", "test_roc_auc_ci_lower", "test_roc_auc_ci_upper",
    "test_pr_auc", "test_pr_auc_ci_lower", "test_pr_auc_ci_upper",
    "test_brier", "test_brier_ci_lower", "test_brier_ci_upper",

    # 各模型 vs model5 的显著性（本行模型 vs model5）
    "test_vs_model5_auroc_pvalue", "test_vs_model5_auprc_pvalue",

    # 仅 model5 vs model4 的 NRI/IDI（总 + 事件组 + 非事件组）
    "test_model5_vs_model4_nri", "test_model5_vs_model4_nri_lower", "test_model5_vs_model4_nri_upper",
    "test_model5_vs_model4_nri_event", "test_model5_vs_model4_nri_event_lower", "test_model5_vs_model4_nri_event_upper",
    "test_model5_vs_model4_nri_nonevent", "test_model5_vs_model4_nri_nonevent_lower", "test_model5_vs_model4_nri_nonevent_upper",
    "test_model5_vs_model4_idi", "test_model5_vs_model4_idi_lower", "test_model5_vs_model4_idi_upper",
]


# ==========================
# 工具函数：DeLong 检验 AUROC
# ==========================

def compute_midrank(x: np.ndarray) -> np.ndarray:
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midrank = 0.5 * (i + j - 1)
        midranks[i:j] = midrank
        i = j
    result = np.empty(n, dtype=float)
    result[sorted_idx] = midranks + 1  # 1-based
    return result


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])

    tz = np.empty((k, m + n))
    for r in range(k):
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n

    return aucs, delong_cov


def calc_pvalue(aucs: np.ndarray, covar: np.ndarray) -> float:
    assert len(aucs) == 2
    l = np.array([[1, -1]])
    diff = np.dot(l, aucs)[0]
    var = np.dot(np.dot(l, covar), l.T)[0, 0]
    z = diff / np.sqrt(var)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return p


def delong_roc_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)
    assert y_true.shape == y_pred1.shape == y_pred2.shape

    order = np.argsort(-y_true)  # y=1 在前
    y_true_sorted = y_true[order]
    preds_sorted = np.vstack([y_pred1[order], y_pred2[order]])

    label_1_count = int(y_true_sorted.sum())
    aucs, delong_cov = fast_delong(preds_sorted, label_1_count)
    p_value = calc_pvalue(aucs, delong_cov)

    return aucs[0], aucs[1], p_value


# ==========================
# 工具函数：metric + bootstrap
# ==========================

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return float(np.mean((y_prob - y_true) **2))


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    point = metric_fn(y_true, y_prob)

    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_bs = y_true[idx]
        p_bs = y_prob[idx]
        if len(np.unique(y_bs)) < 2:
            continue
        samples.append(metric_fn(y_bs, p_bs))
    samples = np.array(samples)
    if len(samples) == 0:
        return point, np.nan, np.nan

    lower = float(np.percentile(samples, 2.5))
    upper = float(np.percentile(samples, 97.5))
    return point, lower, upper


def bootstrap_auprc_difference(
    y_true: np.ndarray,
    y_prob_new: np.ndarray,
    y_prob_old: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE
) -> Tuple[float, float, float, float]:
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_prob_new = np.asarray(y_prob_new)
    y_prob_old = np.asarray(y_prob_old)
    n = len(y_true)

    diff_samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_bs = y_true[idx]
        new_bs = y_prob_new[idx]
        old_bs = y_prob_old[idx]
        if len(np.unique(y_bs)) < 2:
            continue
        auprc_new = average_precision_score(y_bs, new_bs)
        auprc_old = average_precision_score(y_bs, old_bs)
        diff_samples.append(auprc_new - auprc_old)

    diff_samples = np.array(diff_samples)
    if len(diff_samples) == 0:
        return np.nan, np.nan, np.nan, np.nan

    diff_mean = float(np.mean(diff_samples))
    ci_lower = float(np.percentile(diff_samples, 2.5))
    ci_upper = float(np.percentile(diff_samples, 97.5))

    p_left = np.mean(diff_samples <= 0)
    p_right = np.mean(diff_samples >= 0)
    p_value = float(2 * min(p_left, p_right))

    return diff_mean, ci_lower, ci_upper, p_value


# ==========================
# NRI / IDI + bootstrap
# ==========================

def continuous_nri_idi(
    y_true: np.ndarray,
    p_old: np.ndarray,
    p_new: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    返回：
    - nri_event:     事件组 NRI
    - nri_nonevent:  非事件组 NRI
    - nri_total:     总 NRI
    - idi:           IDI
    """
    y_true = np.asarray(y_true)
    p_old = np.asarray(p_old)
    p_new = np.asarray(p_new)

    events = y_true == 1
    nonevents = y_true == 0

    up_event = np.sum((p_new > p_old) & events)
    down_event = np.sum((p_new < p_old) & events)
    up_nonevent = np.sum((p_new > p_old) & nonevents)
    down_nonevent = np.sum((p_new < p_old) & nonevents)

    n_event = max(np.sum(events), 1)
    n_nonevent = max(np.sum(nonevents), 1)

    nri_event = (up_event - down_event) / n_event
    nri_nonevent = (down_nonevent - up_nonevent) / n_nonevent
    nri_total = float(nri_event + nri_nonevent)

    def safe_mean(x):
        return float(np.mean(x)) if len(x) > 0 else 0.0

    old_slope = safe_mean(p_old[events]) - safe_mean(p_old[nonevents])
    new_slope = safe_mean(p_new[events]) - safe_mean(p_new[nonevents])
    idi = float(new_slope - old_slope)

    return float(nri_event), float(nri_nonevent), nri_total, idi


def bootstrap_nri_idi(
    y_true: np.ndarray,
    p_old: np.ndarray,
    p_new: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE
) -> Tuple[float, float, float,
           float, float, float,
           float, float, float,
           float, float, float]:
    """
    返回 12 个值：
    - 总 NRI:          mean, lower, upper
    - 事件组 NRI:      mean, lower, upper
    - 非事件组 NRI:    mean, lower, upper
    - IDI:            mean, lower, upper
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    p_old = np.asarray(p_old)
    p_new = np.asarray(p_new)
    n = len(y_true)

    nri_total_list = []
    nri_event_list = []
    nri_nonevent_list = []
    idi_list = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_bs = y_true[idx]
        old_bs = p_old[idx]
        new_bs = p_new[idx]
        if len(np.unique(y_bs)) < 2:
            continue
        nri_event_bs, nri_nonevent_bs, nri_bs, idi_bs = continuous_nri_idi(y_bs, old_bs, new_bs)
        nri_total_list.append(nri_bs)
        nri_event_list.append(nri_event_bs)
        nri_nonevent_list.append(nri_nonevent_bs)
        idi_list.append(idi_bs)

    if len(nri_total_list) == 0:
        return (np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan)

    nri_total_arr = np.array(nri_total_list)
    nri_event_arr = np.array(nri_event_list)
    nri_nonevent_arr = np.array(nri_nonevent_list)
    idi_arr = np.array(idi_list)

    def mean_ci(arr: np.ndarray) -> Tuple[float, float, float]:
        m = float(np.mean(arr))
        l = float(np.percentile(arr, 2.5))
        u = float(np.percentile(arr, 97.5))
        return m, l, u

    nri_total, nri_total_l, nri_total_u = mean_ci(nri_total_arr)
    nri_event, nri_event_l, nri_event_u = mean_ci(nri_event_arr)
    nri_nonevent, nri_nonevent_l, nri_nonevent_u = mean_ci(nri_nonevent_arr)
    idi, idi_l, idi_u = mean_ci(idi_arr)

    return (nri_total, nri_total_l, nri_total_u,
            nri_event, nri_event_l, nri_event_u,
            nri_nonevent, nri_nonevent_l, nri_nonevent_u,
            idi, idi_l, idi_u)


# ==========================
# 主流程
# ==========================

def main():
    print(f"[INFO] 初始化 summary 表: {SUMMARY_CSV}", flush=True)
    pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(SUMMARY_CSV, index=False)

    for outcome in OUTCOMES:
        print(f"\n[INFO] ===== 开始处理 outcome: {outcome} =====", flush=True)

        cv_data = {}
        test_data = {}

        for model_tag in MODELS_ORDER:
            cv_file = os.path.join(CV_DIR, f"{outcome}_{model_tag}_outer5fold_predictions.csv")
            test_file = os.path.join(TEST_DIR, f"{outcome}_{model_tag}_test_predictions.csv")

            if not os.path.exists(cv_file):
                print(f"[WARN] CV file not found for {outcome}, {model_tag}: {cv_file}", flush=True)
                continue
            if not os.path.exists(test_file):
                print(f"[WARN] Test file not found for {outcome}, {model_tag}: {test_file}", flush=True)
                continue

            print(f"[INFO] 读取文件: {cv_file}", flush=True)
            df_cv = pd.read_csv(cv_file)

            print(f"[INFO] 读取文件: {test_file}", flush=True)
            df_test = pd.read_csv(test_file)

            cv_data[model_tag] = df_cv
            test_data[model_tag] = df_test

        missing_models = [m for m in MODELS_ORDER if m not in cv_data or m not in test_data]
        if missing_models:
            print(f"[WARN] outcome {outcome} 缺失模型: {missing_models}，跳过该 outcome。", flush=True)
            continue

        print(f"[INFO] outcome {outcome} 所有模型文件读取完成。", flush=True)

        # ------------- 计算所有模型 vs model5 的 AUROC/AUPRC p 值 -------------
        cv_vs_model5_auroc_p = {}
        cv_vs_model5_auprc_p = {}
        test_vs_model5_auroc_p = {}
        test_vs_model5_auprc_p = {}

        # 5 vs 4 的 NRI/IDI（只算一次）
        cv_nri_total = cv_nri_total_l = cv_nri_total_u = np.nan
        cv_nri_event = cv_nri_event_l = cv_nri_event_u = np.nan
        cv_nri_nonevent = cv_nri_nonevent_l = cv_nri_nonevent_u = np.nan
        cv_idi = cv_idi_l = cv_idi_u = np.nan

        test_nri_total = test_nri_total_l = test_nri_total_u = np.nan
        test_nri_event = test_nri_event_l = test_nri_event_u = np.nan
        test_nri_nonevent = test_nri_nonevent_l = test_nri_nonevent_u = np.nan
        test_idi = test_idi_l = test_idi_u = np.nan

        for base_tag in ["model1_base", "model2_clinical", "model3_dynamic", "model4_clinical_baseline_exam"]:
            print(f"[INFO] 对齐 {base_tag} vs model5_clinical_dynamic_exam 的样本 (CV & Test) ...", flush=True)

            # CV 对齐
            cv_base = cv_data[base_tag][["eid", "actual", "pred_raw"]].rename(
                columns={"pred_raw": "pred_base", "actual": "actual"}
            )
            cv_m5 = cv_data["model5_clinical_dynamic_exam"][["eid", "pred_raw"]].rename(
                columns={"pred_raw": "pred_m5"}
            )
            cv_merge = pd.merge(cv_base, cv_m5, on="eid", how="inner")

            y_cv_pair = cv_merge["actual"].values
            p_base_cv = cv_merge["pred_base"].values
            p5_cv_pair = cv_merge["pred_m5"].values

            # Test 对齐
            test_base = test_data[base_tag][["eid", "actual", "pred_raw"]].rename(
                columns={"pred_raw": "pred_base", "actual": "actual"}
            )
            test_m5 = test_data["model5_clinical_dynamic_exam"][["eid", "pred_raw"]].rename(
                columns={"pred_raw": "pred_m5"}
            )
            test_merge = pd.merge(test_base, test_m5, on="eid", how="inner")

            y_test_pair = test_merge["actual"].values
            p_base_test = test_merge["pred_base"].values
            p5_test_pair = test_merge["pred_m5"].values

            print(f"[INFO] 对齐完成: {base_tag} vs model5, CV 样本数 {len(y_cv_pair)}, Test 样本数 {len(y_test_pair)}", flush=True)

            # CV AUROC DeLong
            print(f"[INFO] 计算 CV DeLong AUROC: model5 vs {base_tag} ...", flush=True)
            cv_auc_base, cv_auc5, cv_p_auroc = delong_roc_test(y_cv_pair, p_base_cv, p5_cv_pair)
            cv_vs_model5_auroc_p[base_tag] = cv_p_auroc
            print(f"[INFO] CV AUROC: {base_tag}={cv_auc_base:.4f}, model5={cv_auc5:.4f}, p={cv_p_auroc:.4g}", flush=True)

            # CV AUPRC diff
            print(f"[INFO] 计算 CV AUPRC 差异 (bootstrap): model5 - {base_tag} ...", flush=True)
            cv_auprc_diff, _, _, cv_p_auprc = bootstrap_auprc_difference(
                y_cv_pair, p5_cv_pair, p_base_cv, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE
            )
            cv_vs_model5_auprc_p[base_tag] = cv_p_auprc
            print(f"[INFO] CV AUPRC diff={cv_auprc_diff:.4f}, p={cv_p_auprc:.4g}", flush=True)

            # Test AUROC DeLong
            print(f"[INFO] 计算 Test DeLong AUROC: model5 vs {base_tag} ...", flush=True)
            test_auc_base, test_auc5, test_p_auroc = delong_roc_test(y_test_pair, p_base_test, p5_test_pair)
            test_vs_model5_auroc_p[base_tag] = test_p_auroc
            print(f"[INFO] Test AUROC: {base_tag}={test_auc_base:.4f}, model5={test_auc5:.4f}, p={test_p_auroc:.4g}", flush=True)

            # Test AUPRC diff
            print(f"[INFO] 计算 Test AUPRC 差异 (bootstrap): model5 - {base_tag} ...", flush=True)
            test_auprc_diff, _, _, test_p_auprc = bootstrap_auprc_difference(
                y_test_pair, p5_test_pair, p_base_test, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE
            )
            test_vs_model5_auprc_p[base_tag] = test_p_auprc
            print(f"[INFO] Test AUPRC diff={test_auprc_diff:.4f}, p={test_p_auprc:.4g}", flush=True)

            # 只有 base_tag == model4_clinical_baseline_exam 时计算 NRI / IDI
            if base_tag == "model4_clinical_baseline_exam":
                print(f"[INFO] 计算 CV NRI / IDI (bootstrap): model5 vs model4 ...", flush=True)
                (cv_nri_total, cv_nri_total_l, cv_nri_total_u,
                 cv_nri_event, cv_nri_event_l, cv_nri_event_u,
                 cv_nri_nonevent, cv_nri_nonevent_l, cv_nri_nonevent_u,
                 cv_idi, cv_idi_l, cv_idi_u) = bootstrap_nri_idi(
                    y_cv_pair, p_base_cv, p5_cv_pair, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE
                )
                print(
                    f"[INFO] CV NRI_total={cv_nri_total:.4f} [{cv_nri_total_l:.4f}, {cv_nri_total_u:.4f}], "
                    f"NRI_event={cv_nri_event:.4f} [{cv_nri_event_l:.4f}, {cv_nri_event_u:.4f}], "
                    f"NRI_nonevent={cv_nri_nonevent:.4f} [{cv_nri_nonevent_l:.4f}, {cv_nri_nonevent_u:.4f}], "
                    f"IDI={cv_idi:.4f} [{cv_idi_l:.4f}, {cv_idi_u:.4f}]",
                    flush=True,
                )

                print(f"[INFO] 计算 Test NRI / IDI (bootstrap): model5 vs model4 ...", flush=True)
                (test_nri_total, test_nri_total_l, test_nri_total_u,
                 test_nri_event, test_nri_event_l, test_nri_event_u,
                 test_nri_nonevent, test_nri_nonevent_l, test_nri_nonevent_u,
                 test_idi, test_idi_l, test_idi_u) = bootstrap_nri_idi(
                    y_test_pair, p_base_test, p5_test_pair, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE
                )
                print(
                    f"[INFO] Test NRI_total={test_nri_total:.4f} [{test_nri_total_l:.4f}, {test_nri_total_u:.4f}], "
                    f"NRI_event={test_nri_event:.4f} [{test_nri_event_l:.4f}, {test_nri_event_u:.4f}], "
                    f"NRI_nonevent={test_nri_nonevent:.4f} [{test_nri_nonevent_l:.4f}, {test_nri_nonevent_u:.4f}], "
                    f"IDI={test_idi:.4f} [{test_idi_l:.4f}, {test_idi_u:.4f}]",
                    flush=True,
                )

        # 每个模型：CV / Test 指标 + 流式写 summary
        for model_tag in MODELS_ORDER:
            print(f"[INFO] 计算 outcome={outcome}, model={model_tag} 的指标 ...", flush=True)
            df_cv = cv_data[model_tag]
            df_test = test_data[model_tag]

            y_cv_all = df_cv["actual"].values
            p_cv_all = df_cv["pred_raw"].values

            y_test_all = df_test["actual"].values
            p_test_all = df_test["pred_raw"].values

            cv_auc_mean, cv_auc_l, cv_auc_u = bootstrap_metric_ci(
                y_cv_all, p_cv_all, roc_auc_score
            )
            cv_pr_mean, cv_pr_l, cv_pr_u = bootstrap_metric_ci(
                y_cv_all, p_cv_all, average_precision_score
            )
            cv_brier_mean, cv_brier_l, cv_brier_u = bootstrap_metric_ci(
                y_cv_all, p_cv_all, brier_score
            )

            test_auc_mean, test_auc_l, test_auc_u = bootstrap_metric_ci(
                y_test_all, p_test_all, roc_auc_score
            )
            test_pr_mean, test_pr_l, test_pr_u = bootstrap_metric_ci(
                y_test_all, p_test_all, average_precision_score
            )
            test_brier_mean, test_brier_l, test_brier_u = bootstrap_metric_ci(
                y_test_all, p_test_all, brier_score
            )

            # 各模型 vs model5 的 p 值：model5 行为 NaN，其它行从字典取
            if model_tag == "model5_clinical_dynamic_exam":
                cv_vs5_auroc_field = np.nan
                cv_vs5_auprc_field = np.nan
                test_vs5_auroc_field = np.nan
                test_vs5_auprc_field = np.nan
            else:
                cv_vs5_auroc_field = cv_vs_model5_auroc_p.get(model_tag, np.nan)
                cv_vs5_auprc_field = cv_vs_model5_auprc_p.get(model_tag, np.nan)
                test_vs5_auroc_field = test_vs_model5_auroc_p.get(model_tag, np.nan)
                test_vs5_auprc_field = test_vs_model5_auprc_p.get(model_tag, np.nan)

            # NRI/IDI 只针对 model5 vs model4，并且在 model4 & model5 两行都填值，其它模型 NaN
            if model_tag in ["model4_clinical_baseline_exam", "model5_clinical_dynamic_exam"]:
                cv_nri_total_field = cv_nri_total
                cv_nri_total_l_field = cv_nri_total_l
                cv_nri_total_u_field = cv_nri_total_u
                cv_nri_event_field = cv_nri_event
                cv_nri_event_l_field = cv_nri_event_l
                cv_nri_event_u_field = cv_nri_event_u
                cv_nri_nonevent_field = cv_nri_nonevent
                cv_nri_nonevent_l_field = cv_nri_nonevent_l
                cv_nri_nonevent_u_field = cv_nri_nonevent_u
                cv_idi_field = cv_idi
                cv_idi_l_field = cv_idi_l
                cv_idi_u_field = cv_idi_u

                test_nri_total_field = test_nri_total
                test_nri_total_l_field = test_nri_total_l
                test_nri_total_u_field = test_nri_total_u
                test_nri_event_field = test_nri_event
                test_nri_event_l_field = test_nri_event_l
                test_nri_event_u_field = test_nri_event_u
                test_nri_nonevent_field = test_nri_nonevent
                test_nri_nonevent_l_field = test_nri_nonevent_l
                test_nri_nonevent_u_field = test_nri_nonevent_u
                test_idi_field = test_idi
                test_idi_l_field = test_idi_l
                test_idi_u_field = test_idi_u
            else:
                cv_nri_total_field = np.nan
                cv_nri_total_l_field = np.nan
                cv_nri_total_u_field = np.nan
                cv_nri_event_field = np.nan
                cv_nri_event_l_field = np.nan
                cv_nri_event_u_field = np.nan
                cv_nri_nonevent_field = np.nan
                cv_nri_nonevent_l_field = np.nan
                cv_nri_nonevent_u_field = np.nan
                cv_idi_field = np.nan
                cv_idi_l_field = np.nan
                cv_idi_u_field = np.nan

                test_nri_total_field = np.nan
                test_nri_total_l_field = np.nan
                test_nri_total_u_field = np.nan
                test_nri_event_field = np.nan
                test_nri_event_l_field = np.nan
                test_nri_event_u_field = np.nan
                test_nri_nonevent_field = np.nan
                test_nri_nonevent_l_field = np.nan
                test_nri_nonevent_u_field = np.nan
                test_idi_field = np.nan
                test_idi_l_field = np.nan
                test_idi_u_field = np.nan

            row_dict = {
                "outcome": outcome,
                "model_type": model_tag,
                "cv_roc_auc_mean": cv_auc_mean,
                "cv_roc_auc_ci_lower": cv_auc_l,
                "cv_roc_auc_ci_upper": cv_auc_u,
                "cv_pr_auc_mean": cv_pr_mean,
                "cv_pr_auc_ci_lower": cv_pr_l,
                "cv_pr_auc_ci_upper": cv_pr_u,
                "cv_brier_mean": cv_brier_mean,
                "cv_brier_ci_lower": cv_brier_l,
                "cv_brier_ci_upper": cv_brier_u,

                "cv_vs_model5_auroc_pvalue": cv_vs5_auroc_field,
                "cv_vs_model5_auprc_pvalue": cv_vs5_auprc_field,

                "cv_model5_vs_model4_nri": cv_nri_total_field,
                "cv_model5_vs_model4_nri_lower": cv_nri_total_l_field,
                "cv_model5_vs_model4_nri_upper": cv_nri_total_u_field,
                "cv_model5_vs_model4_nri_event": cv_nri_event_field,
                "cv_model5_vs_model4_nri_event_lower": cv_nri_event_l_field,
                "cv_model5_vs_model4_nri_event_upper": cv_nri_event_u_field,
                "cv_model5_vs_model4_nri_nonevent": cv_nri_nonevent_field,
                "cv_model5_vs_model4_nri_nonevent_lower": cv_nri_nonevent_l_field,
                "cv_model5_vs_model4_nri_nonevent_upper": cv_nri_nonevent_u_field,
                "cv_model5_vs_model4_idi": cv_idi_field,
                "cv_model5_vs_model4_idi_lower": cv_idi_l_field,
                "cv_model5_vs_model4_idi_upper": cv_idi_u_field,

                "test_roc_auc": test_auc_mean,
                "test_roc_auc_ci_lower": test_auc_l,
                "test_roc_auc_ci_upper": test_auc_u,
                "test_pr_auc": test_pr_mean,
                "test_pr_auc_ci_lower": test_pr_l,
                "test_pr_auc_ci_upper": test_pr_u,
                "test_brier": test_brier_mean,
                "test_brier_ci_lower": test_brier_l,
                "test_brier_ci_upper": test_brier_u,

                "test_vs_model5_auroc_pvalue": test_vs5_auroc_field,
                "test_vs_model5_auprc_pvalue": test_vs5_auprc_field,

                "test_model5_vs_model4_nri": test_nri_total_field,
                "test_model5_vs_model4_nri_lower": test_nri_total_l_field,
                "test_model5_vs_model4_nri_upper": test_nri_total_u_field,
                "test_model5_vs_model4_nri_event": test_nri_event_field,
                "test_model5_vs_model4_nri_event_lower": test_nri_event_l_field,
                "test_model5_vs_model4_nri_event_upper": test_nri_event_u_field,
                "test_model5_vs_model4_nri_nonevent": test_nri_nonevent_field,
                "test_model5_vs_model4_nri_nonevent_lower": test_nri_nonevent_l_field,
                "test_model5_vs_model4_nri_nonevent_upper": test_nri_nonevent_u_field,
                "test_model5_vs_model4_idi": test_idi_field,
                "test_model5_vs_model4_idi_lower": test_idi_l_field,
                "test_model5_vs_model4_idi_upper": test_idi_u_field,
            }

            df_row = pd.DataFrame([row_dict], columns=SUMMARY_COLUMNS)
            df_row.to_csv(SUMMARY_CSV, mode="a", header=False, index=False)
            print(f"[INFO] 已写入 summary 行: outcome={outcome}, model={model_tag}", flush=True)

        print(f"[INFO] ===== outcome {outcome} 处理完成 =====", flush=True)

    print(f"\n[INFO] 全部处理完成，summary 已流式写入: {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
