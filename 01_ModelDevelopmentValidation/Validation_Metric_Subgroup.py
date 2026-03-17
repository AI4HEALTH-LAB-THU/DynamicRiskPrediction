import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# ========= 从主分析脚本 Metric_full.py 中复用配置和函数 =========
from Metric_full import (
    BASE_DIR,
    CV_DIR,
    OUTCOMES,
    MODELS_ORDER,
    N_BOOTSTRAP,
    RANDOM_STATE,
    brier_score,
    bootstrap_metric_ci,
)

# ========= 配置区域 =========

# 亚组信息文件
SUBGROUP_FILE = "data/train_test_dataset/dat_main_subgroup.feather"

# 需要分析的亚组变量
SUBGROUP_VARS = ["city_raw", "age_group", "sex", "hypertension"]

# 只分析两个模型
BASE_MODEL_TAG = "model4_clinical_baseline_exam"
NEW_MODEL_TAG = "model5_clinical_dynamic_exam"
MODELS_FOR_SUBGROUP = [BASE_MODEL_TAG, NEW_MODEL_TAG]

# outcome
OUTCOMES_SUBSET = [
    "mi",
    "afib_flutter",
    "cor_pulmonale",
    "chf",
    "stroke",
    "ischemic_stroke",
    "hemorrhagic_stroke",
    "arterial_disease",
    "copd",
    "liver_fibrosis_cirrhosis",
    "liver_failure",
    "renal_failure",
    "diabetes",
    "thyroid_disease",
    "parkinson",
    "dementia",
    "cancer_all",
    "liver_cancer",
    "lung_cancer",
    "kidney_cancer"
]

# 输出目录
SUBGROUP_OUT_DIR = os.path.join(BASE_DIR, "subgroup_analysis")
os.makedirs(SUBGROUP_OUT_DIR, exist_ok=True)

# 各亚组性能 (AUROC / AUPRC / Brier + 全局 p)
SUBGROUP_METRIC_CSV = os.path.join(SUBGROUP_OUT_DIR, "subgroup_performance.csv")

# 各亚组 NRI / IDI (NEW vs BASE)
SUBGROUP_NRI_CSV = os.path.join(SUBGROUP_OUT_DIR, "subgroup_nri_idi.csv")

# 各亚组 AUROC / AUPRC 差值 (NEW - BASE) + p 值
SUBGROUP_ROCPR_DIFF_CSV = os.path.join(SUBGROUP_OUT_DIR, "subgroup_roc_pr_diff.csv")


# ========= 工具函数：连续型 NRI / IDI =========

def _compute_nri_idi_once(y_true, p_old, p_new):
    """
    连续型 NRI & IDI 的一次性计算（不带 bootstrap）。
    y_true: 0/1
    p_old, p_new: 两个模型的风险预测

    返回:
        nri, nri_events, nri_nonevents, idi  (float)
    """
    y_true = np.asarray(y_true)
    p_old = np.asarray(p_old)
    p_new = np.asarray(p_new)

    events = y_true == 1
    non_events = y_true == 0

    if events.sum() == 0 or non_events.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Continuous NRI
    # 对事件：新模型风险上升视为 positive，下降视为 negative
    up_events = np.mean(p_new[events] > p_old[events])
    down_events = np.mean(p_new[events] < p_old[events])
    nri_events = up_events - down_events

    # 对非事件：新模型风险下降视为 positive，上升视为 negative
    up_nonevents = np.mean(p_new[non_events] > p_old[non_events])
    down_nonevents = np.mean(p_new[non_events] < p_old[non_events])
    nri_nonevents = down_nonevents - up_nonevents

    nri = nri_events + nri_nonevents

    # IDI (difference in discrimination slopes)
    # slope = mean(p | event) - mean(p | non-event)
    slope_old = p_old[events].mean() - p_old[non_events].mean()
    slope_new = p_new[events].mean() - p_new[non_events].mean()
    idi = slope_new - slope_old

    return float(nri), float(nri_events), float(nri_nonevents), float(idi)


def bootstrap_nri_idi(
    y_true,
    p_old,
    p_new,
    n_bootstrap=N_BOOTSTRAP,
    random_state=RANDOM_STATE,
):
    """
    对 NRI / IDI 做 bootstrap，给出点估计和 95% CI。
    返回:
        nri, nri_l, nri_u, 
        nri_events, nri_events_l, nri_events_u,
        nri_nonevents, nri_nonevents_l, nri_nonevents_u,
        idi, idi_l, idi_u
    """
    y_true = np.asarray(y_true)
    p_old = np.asarray(p_old)
    p_new = np.asarray(p_new)

    if len(np.unique(y_true)) < 2:
        return (np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan)

    nri_point, nri_events_point, nri_nonevents_point, idi_point = _compute_nri_idi_once(y_true, p_old, p_new)

    rng = np.random.default_rng(random_state)
    nri_samples = []
    nri_events_samples = []
    nri_nonevents_samples = []
    idi_samples = []

    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_bs = y_true[idx]
        p_old_bs = p_old[idx]
        p_new_bs = p_new[idx]

        if len(np.unique(y_bs)) < 2:
            continue

        nri_bs, nri_events_bs, nri_nonevents_bs, idi_bs = _compute_nri_idi_once(y_bs, p_old_bs, p_new_bs)
        if np.isnan(nri_bs) or np.isnan(idi_bs):
            continue

        nri_samples.append(nri_bs)
        nri_events_samples.append(nri_events_bs)
        nri_nonevents_samples.append(nri_nonevents_bs)
        idi_samples.append(idi_bs)

    if len(nri_samples) == 0:
        return (nri_point, np.nan, np.nan,
                nri_events_point, np.nan, np.nan,
                nri_nonevents_point, np.nan, np.nan,
                idi_point, np.nan, np.nan)

    nri_samples = np.asarray(nri_samples)
    nri_events_samples = np.asarray(nri_events_samples)
    nri_nonevents_samples = np.asarray(nri_nonevents_samples)
    idi_samples = np.asarray(idi_samples)

    nri_l = float(np.percentile(nri_samples, 2.5))
    nri_u = float(np.percentile(nri_samples, 97.5))
    nri_events_l = float(np.percentile(nri_events_samples, 2.5))
    nri_events_u = float(np.percentile(nri_events_samples, 97.5))
    nri_nonevents_l = float(np.percentile(nri_nonevents_samples, 2.5))
    nri_nonevents_u = float(np.percentile(nri_nonevents_samples, 97.5))
    idi_l = float(np.percentile(idi_samples, 2.5))
    idi_u = float(np.percentile(idi_samples, 97.5))

    return (nri_point, nri_l, nri_u,
            nri_events_point, nri_events_l, nri_events_u,
            nri_nonevents_point, nri_nonevents_l, nri_nonevents_u,
            idi_point, idi_l, idi_u)


# ========= 工具函数：全局差异检验（亚组整体是否有差异） =========

def permutation_global_test(
    y_true,
    y_prob,
    group,
    metric_fn,
    n_perm=N_BOOTSTRAP,
    random_state=RANDOM_STATE,
    require_two_classes=True,
):
    """
    检验：同一模型在不同亚组上的某个指标（例如 AUROC）
    是否整体存在差异（全局 p 值）。

    统计量 T：各组指标的“样本数加权方差”
        T = sum_i n_i * (m_i - m_bar)^2
      n_i: 第 i 组样本数
      m_i: 第 i 组的 metric
      m_bar: 样本数加权平均

    置换原理：
      - 随机打乱 group 标签，重算 T，得到零分布
      - p = P(T_perm >= T_obs) 右尾检验
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    group = np.asarray(group)

    # 去掉 group 缺失值
    mask = ~pd.isna(group)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    group = group[mask]

    unique_groups = np.unique(group)
    if unique_groups.size < 2:
        return np.nan

    def compute_stat(y, p, g):
        metrics = []
        ns = []
        for lvl in unique_groups:
            m_mask = g == lvl
            if not np.any(m_mask):
                continue

            y_g = y[m_mask]
            p_g = p[m_mask]
            n_g = len(y_g)

            if require_two_classes and len(np.unique(y_g)) < 2:
                return None

            try:
                m = float(metric_fn(y_g, p_g))
            except Exception:
                return None

            metrics.append(m)
            ns.append(n_g)

        if len(metrics) < 2:
            return None

        metrics = np.asarray(metrics)
        ns = np.asarray(ns, dtype=float)
        m_bar = np.sum(ns * metrics) / np.sum(ns)
        stat = float(np.sum(ns * (metrics - m_bar) ** 2))
        return stat

    stat_obs = compute_stat(y_true, y_prob, group)
    if stat_obs is None:
        return np.nan

    rng = np.random.default_rng(random_state)
    stats_perm = []

    for _ in range(n_perm):
        g_perm = rng.permutation(group)
        stat_perm = compute_stat(y_true, y_prob, g_perm)
        if stat_perm is None:
            continue
        stats_perm.append(stat_perm)

    if len(stats_perm) == 0:
        return np.nan

    stats_perm = np.asarray(stats_perm)
    p_val = (1.0 + np.sum(stats_perm >= stat_obs)) / (1.0 + len(stats_perm))
    return float(min(p_val, 1.0))


# ========= 工具函数：比较两个模型在同一亚组的 AUROC / AUPRC 差值 =========

def bootstrap_metric_diff(
    y_true,
    p_old,
    p_new,
    metric_fn,
    n_bootstrap=N_BOOTSTRAP,
    random_state=RANDOM_STATE,
):
    """
    比较两个模型（p_new vs p_old）在同一亚组中某个 metric（AUROC 或 AUPRC）
    的差值及其 95% CI 和 p 值。

    返回：
        diff_point, ci_l, ci_u, p_val
      其中 diff_point = metric_new - metric_old
    """
    y_true = np.asarray(y_true)
    p_old = np.asarray(p_old)
    p_new = np.asarray(p_new)

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan, np.nan

    try:
        m_old = float(metric_fn(y_true, p_old))
        m_new = float(metric_fn(y_true, p_new))
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

    diff_point = m_new - m_old

    rng = np.random.default_rng(random_state)
    diff_samples = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_bs = y_true[idx]
        p_old_bs = p_old[idx]
        p_new_bs = p_new[idx]

        if len(np.unique(y_bs)) < 2:
            continue

        try:
            d = float(metric_fn(y_bs, p_new_bs) - metric_fn(y_bs, p_old_bs))
        except Exception:
            continue

        diff_samples.append(d)

    if len(diff_samples) == 0:
        return diff_point, np.nan, np.nan, np.nan

    diff_samples = np.asarray(diff_samples)
    ci_l = float(np.percentile(diff_samples, 2.5))
    ci_u = float(np.percentile(diff_samples, 97.5))

    # 双侧 p 值：围绕 0 的对称性
    p_left = np.mean(diff_samples <= 0.0)
    p_right = np.mean(diff_samples >= 0.0)
    p_val = float(2 * min(p_left, p_right))
    p_val = min(p_val, 1.0)

    return diff_point, ci_l, ci_u, p_val


# ========= 计算并保存亚组 NRI / IDI =========

def compute_and_save_subgroup_nri_idi(outcome, cv_data):
    """
    对给定 outcome，使用 cv_data 中的 BASE_MODEL_TAG 和 NEW_MODEL_TAG，
    在每个 subgroup_var 和 每个 subgroup_level 下计算：
        NRI / IDI 以及各自 95% CI，
        包括事件组和非事件组的 NRI 及其 95% CI
    并将结果 append 到 SUBGROUP_NRI_CSV。
    """
    if BASE_MODEL_TAG not in cv_data or NEW_MODEL_TAG not in cv_data:
        print(
            f"[WARN] outcome={outcome} 缺少 {BASE_MODEL_TAG} 或 {NEW_MODEL_TAG}，"
            f"无法计算 NRI/IDI。",
            flush=True,
        )
        return

    df_base = cv_data[BASE_MODEL_TAG].copy()
    df_new = cv_data[NEW_MODEL_TAG].copy()

    df_base = df_base[
        ["eid", "actual", "pred_raw"] +
        [v for v in SUBGROUP_VARS if v in df_base.columns]
    ].rename(columns={"pred_raw": "pred_base"})

    df_new = df_new[["eid", "pred_raw"]].rename(columns={"pred_raw": "pred_new"})

    df_merge = pd.merge(df_base, df_new, on="eid", how="inner")

    rows = []

    for sg_var in SUBGROUP_VARS:
        if sg_var not in df_merge.columns:
            print(
                f"[WARN] outcome={outcome} NRI/IDI: {sg_var} 不在数据中，跳过该亚组变量。",
                flush=True,
            )
            continue

        levels = df_merge[sg_var].dropna().unique().tolist()

        for lvl in levels:
            df_g = df_merge[df_merge[sg_var] == lvl]

            y_g = df_g["actual"].values
            p_old_g = df_g["pred_base"].values
            p_new_g = df_g["pred_new"].values

            n_g = len(df_g)
            n_events_g = int(df_g["actual"].sum())

            if len(np.unique(y_g)) < 2:
                nri = nri_l = nri_u = np.nan
                nri_events = nri_events_l = nri_events_u = np.nan
                nri_nonevents = nri_nonevents_l = nri_nonevents_u = np.nan
                idi = idi_l = idi_u = np.nan
            else:
                (nri, nri_l, nri_u,
                 nri_events, nri_events_l, nri_events_u,
                 nri_nonevents, nri_nonevents_l, nri_nonevents_u,
                 idi, idi_l, idi_u) = bootstrap_nri_idi(
                    y_g, p_old_g, p_new_g, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE
                )

            rows.append(
                {
                    "outcome": outcome,
                    "subgroup_var": sg_var,
                    "subgroup_level": lvl,
                    "n": n_g,
                    "n_events": n_events_g,
                    "nri_new_vs_base": nri,
                    "nri_ci_lower": nri_l,
                    "nri_ci_upper": nri_u,
                    "nri_events": nri_events,
                    "nri_events_ci_lower": nri_events_l,
                    "nri_events_ci_upper": nri_events_u,
                    "nri_nonevents": nri_nonevents,
                    "nri_nonevents_ci_lower": nri_nonevents_l,
                    "nri_nonevents_ci_upper": nri_nonevents_u,
                    "idi_new_vs_base": idi,
                    "idi_ci_lower": idi_l,
                    "idi_ci_upper": idi_u,
                }
            )

    if rows:
        df_rows = pd.DataFrame(rows)
        df_rows.to_csv(
            SUBGROUP_NRI_CSV,
            mode="a",
            header=False,
            index=False,
        )
        print(
            f"[INFO] outcome={outcome} 的亚组 NRI/IDI 已写入 {SUBGROUP_NRI_CSV}",
            flush=True,
        )


# ========= 主流程 =========

def main():
    print(f"[INFO] 读取亚组信息: {SUBGROUP_FILE}", flush=True)
    df_sub = pd.read_feather(SUBGROUP_FILE)

    if df_sub["eid"].duplicated().any():
        raise ValueError("亚组文件中 eid 存在重复，请先去重。")

    print(f"[INFO] 亚组样本数: {len(df_sub)}", flush=True)

    # 初始化三个输出文件（覆盖旧文件）
    # 1) 各模型在各亚组上的 AUROC/AUPRC/Brier + 全局 p
    metric_cols = [
        "outcome",
        "model_type",
        "subgroup_var",
        "subgroup_level",
        "n",
        "n_events",
        "roc_auc",
        "roc_auc_ci_lower",
        "roc_auc_ci_upper",
        "pr_auc",
        "pr_auc_ci_lower",
        "pr_auc_ci_upper",
        "brier",
        "brier_ci_lower",
        "brier_ci_upper",
        "roc_auc_global_p",
        "pr_auc_global_p",
        "brier_global_p",
    ]
    pd.DataFrame(columns=metric_cols).to_csv(SUBGROUP_METRIC_CSV, index=False)

    # 2) 各亚组 NRI / IDI（NEW vs BASE）
    nri_cols = [
        "outcome",
        "subgroup_var",
        "subgroup_level",
        "n",
        "n_events",
        "nri_new_vs_base",
        "nri_ci_lower",
        "nri_ci_upper",
        "nri_events",
        "nri_events_ci_lower",
        "nri_events_ci_upper",
        "nri_nonevents",
        "nri_nonevents_ci_lower",
        "nri_nonevents_ci_upper",
        "idi_new_vs_base",
        "idi_ci_lower",
        "idi_ci_upper",
    ]
    pd.DataFrame(columns=nri_cols).to_csv(SUBGROUP_NRI_CSV, index=False)

    # 3) 各亚组 AUROC / AUPRC 差值（NEW - BASE）+ p 值
    diff_cols = [
        "outcome",
        "subgroup_var",
        "subgroup_level",
        "roc_diff_new_minus_base",
        "roc_ci_lower",
        "roc_ci_upper",
        "roc_p_value",
        "pr_diff_new_minus_base",
        "pr_ci_lower",
        "pr_ci_upper",
        "pr_p_value",
    ]
    pd.DataFrame(columns=diff_cols).to_csv(SUBGROUP_ROCPR_DIFF_CSV, index=False)

    # ------- 只循环 OUTCOMES_SUBSET 中指定的疾病 -------
    for outcome in OUTCOMES_SUBSET:
        print(f"\n[INFO] ===== 开始处理 outcome: {outcome} =====", flush=True)

        # 只读取 model4 / model5 的 CV 预测，并 merge 亚组信息
        cv_data = {}
        for model_tag in MODELS_FOR_SUBGROUP:
            cv_file = os.path.join(
                CV_DIR, f"{outcome}_{model_tag}_outer5fold_predictions.csv"
            )
            if not os.path.exists(cv_file):
                print(f"[WARN] CV file not found: {cv_file}，跳过该模型。", flush=True)
                continue

            print(f"[INFO] 读取 CV 预测文件: {cv_file}", flush=True)
            df_cv = pd.read_csv(cv_file)
            df_cv = df_cv.merge(df_sub, on="eid", how="inner")
            cv_data[model_tag] = df_cv

        if len(cv_data) == 0:
            print(f"[WARN] outcome={outcome} 没有可用的 CV 预测，跳过。", flush=True)
            continue

        # ------- 1) 各模型在各亚组内：AUROC/AUPRC/Brier + 全局 p -------
        for model_tag, df_cv in cv_data.items():
            print(
                f"[INFO] 计算 CV 指标: outcome={outcome}, model={model_tag}",
                flush=True,
            )

            y_all = df_cv["actual"].values
            p_all = df_cv["pred_raw"].values

            for sg_var in SUBGROUP_VARS:
                if sg_var not in df_cv.columns:
                    print(
                        f"[WARN] {sg_var} 不在数据列中，跳过该亚组变量。",
                        flush=True,
                    )
                    continue

                print(
                    f"[INFO] 计算亚组变量 {sg_var} 的性能和全局差异检验...",
                    flush=True,
                )

                g_all = df_cv[sg_var].values

                # 全局 p 值
                p_global_roc = permutation_global_test(
                    y_all,
                    p_all,
                    g_all,
                    metric_fn=roc_auc_score,
                    n_perm=N_BOOTSTRAP,
                    random_state=RANDOM_STATE,
                    require_two_classes=True,
                )
                p_global_pr = permutation_global_test(
                    y_all,
                    p_all,
                    g_all,
                    metric_fn=average_precision_score,
                    n_perm=N_BOOTSTRAP,
                    random_state=RANDOM_STATE,
                    require_two_classes=True,
                )
                p_global_brier = permutation_global_test(
                    y_all,
                    p_all,
                    g_all,
                    metric_fn=brier_score,
                    n_perm=N_BOOTSTRAP,
                    random_state=RANDOM_STATE,
                    require_two_classes=False,
                )

                levels = df_cv[sg_var].dropna().unique().tolist()
                metric_rows = []

                for lvl in levels:
                    df_g = df_cv[df_cv[sg_var] == lvl]
                    y_g = df_g["actual"].values
                    p_g = df_g["pred_raw"].values

                    n_g = len(df_g)
                    n_events_g = int(df_g["actual"].sum())

                    # AUROC / AUPRC
                    if len(np.unique(y_g)) < 2:
                        roc_mean = roc_l = roc_u = np.nan
                        pr_mean = pr_l = pr_u = np.nan
                    else:
                        roc_mean, roc_l, roc_u = bootstrap_metric_ci(
                            y_g, p_g, roc_auc_score
                        )
                        pr_mean, pr_l, pr_u = bootstrap_metric_ci(
                            y_g, p_g, average_precision_score
                        )

                    # Brier
                    brier_mean, brier_l, brier_u = bootstrap_metric_ci(
                        y_g, p_g, brier_score
                    )

                    metric_rows.append(
                        {
                            "outcome": outcome,
                            "model_type": model_tag,
                            "subgroup_var": sg_var,
                            "subgroup_level": lvl,
                            "n": n_g,
                            "n_events": n_events_g,
                            "roc_auc": roc_mean,
                            "roc_auc_ci_lower": roc_l,
                            "roc_auc_ci_upper": roc_u,
                            "pr_auc": pr_mean,
                            "pr_auc_ci_lower": pr_l,
                            "pr_auc_ci_upper": pr_u,
                            "brier": brier_mean,
                            "brier_ci_lower": brier_l,
                            "brier_ci_upper": brier_u,
                            "roc_auc_global_p": p_global_roc,
                            "pr_auc_global_p": p_global_pr,
                            "brier_global_p": p_global_brier,
                        }
                    )

                if metric_rows:
                    pd.DataFrame(metric_rows).to_csv(
                        SUBGROUP_METRIC_CSV,
                        mode="a",
                        header=False,
                        index=False,
                    )

        # ------- 2) 各亚组 NRI / IDI（NEW vs BASE） -------
        compute_and_save_subgroup_nri_idi(outcome, cv_data)

        # ------- 3) 各亚组 AUROC / AUPRC 差值（NEW - BASE） + p 值 -------
        if BASE_MODEL_TAG in cv_data and NEW_MODEL_TAG in cv_data:
            print(
                f"[INFO] 计算 outcome={outcome} 的 AUROC / AUPRC 差值 ("
                f"{NEW_MODEL_TAG} - {BASE_MODEL_TAG})",
                flush=True,
            )

            df_base = cv_data[BASE_MODEL_TAG].rename(columns={"pred_raw": "pred_base"})
            df_new = cv_data[NEW_MODEL_TAG].rename(columns={"pred_raw": "pred_new"})

            df_merge = df_base.merge(
                df_new[["eid", "pred_new"]],
                on="eid",
                how="inner",
            )

            diff_rows = []

            for sg_var in SUBGROUP_VARS:
                if sg_var not in df_merge.columns:
                    print(
                        f"[WARN] {sg_var} 不在数据列中，跳过 AUROC/AUPRC 差值检验。",
                        flush=True,
                    )
                    continue

                for lvl in df_merge[sg_var].dropna().unique().tolist():
                    df_g = df_merge[df_merge[sg_var] == lvl]

                    y_g = df_g["actual"].values
                    p_old_g = df_g["pred_base"].values
                    p_new_g = df_g["pred_new"].values

                    # ROC 差值
                    roc_d, roc_l, roc_u, roc_p = bootstrap_metric_diff(
                        y_g, p_old_g, p_new_g, roc_auc_score
                    )
                    # PR 差值
                    pr_d, pr_l, pr_u, pr_p = bootstrap_metric_diff(
                        y_g, p_old_g, p_new_g, average_precision_score
                    )

                    diff_rows.append(
                        {
                            "outcome": outcome,
                            "subgroup_var": sg_var,
                            "subgroup_level": lvl,
                            "roc_diff_new_minus_base": roc_d,
                            "roc_ci_lower": roc_l,
                            "roc_ci_upper": roc_u,
                            "roc_p_value": roc_p,
                            "pr_diff_new_minus_base": pr_d,
                            "pr_ci_lower": pr_l,
                            "pr_ci_upper": pr_u,
                            "pr_p_value": pr_p,
                        }
                    )

            if diff_rows:
                pd.DataFrame(diff_rows).to_csv(
                    SUBGROUP_ROCPR_DIFF_CSV,
                    mode="a",
                    header=False,
                    index=False,
                )
                print(
                    f"[INFO] outcome={outcome} 的 AUROC/AUPRC 差值检验结果已写入 "
                    f"{SUBGROUP_ROCPR_DIFF_CSV}",
                    flush=True,
                )
        else:
            print(
                f"[WARN] outcome={outcome} 不同时包含 {BASE_MODEL_TAG} 和 "
                f"{NEW_MODEL_TAG}，跳过 AUROC/AUPRC 差值检验。",
                flush=True,
            )

        print(f"[INFO] outcome={outcome} 处理完成。", flush=True)

    print(
        "\n[INFO] 全部亚组分析完成，结果文件：\n"
        f"  1) 性能指标: {SUBGROUP_METRIC_CSV}\n"
        f"  2) NRI / IDI: {SUBGROUP_NRI_CSV}\n"
        f"  3) AUROC/AUPRC 差值: {SUBGROUP_ROCPR_DIFF_CSV}",
        flush=True,
    )


if __name__ == "__main__":
    main()
