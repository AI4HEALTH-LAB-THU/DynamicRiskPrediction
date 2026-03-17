import os
import numpy as np
import pandas as pd

# ==========================
# 全局配置
# ==========================

BASE_DIR = "result/bench_dynamic/xgb/prediction_results/251209_final"
CV_DIR = os.path.join(BASE_DIR, "outer_cv_predictions")
TEST_DIR = os.path.join(BASE_DIR, "external_val_predictions")

DCA_BASE_DIR = os.path.join(BASE_DIR, "dca_outputs")
DCA_CV_DIR = os.path.join(DCA_BASE_DIR, "cv")
DCA_TEST_DIR = os.path.join(DCA_BASE_DIR, "test")
os.makedirs(DCA_CV_DIR, exist_ok=True)
os.makedirs(DCA_TEST_DIR, exist_ok=True)

RANDOM_STATE = 42

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

DCA_MODELS = [
    "model1_base",
    "model2_clinical",
    "model4_clinical_baseline_exam",
    "model5_clinical_dynamic_exam"
]

# ==========================
# DCA 核心函数：同时返回 NB 与 sNB
# ==========================

def _validate_thresholds(thresholds: np.ndarray) -> np.ndarray:
    thr = np.asarray(thresholds, dtype=float)
    # 避免 t=1 导致除零
    thr = np.clip(thr, 0.0, 1.0 - 1e-12)
    return thr


def decision_curve_nb_and_snb(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray
):
    """
    对给定模型预测概率，计算每个阈值下：
    NB(t)  = TP/N - FP/N * t/(1-t)
    sNB(t) = NB(t) / prevalence
    返回：NB_array, sNB_array, prevalence
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    thr = _validate_thresholds(thresholds)

    N = len(y_true)
    prevalence = float(np.mean(y_true)) if N > 0 else 0.0

    NB = np.zeros_like(thr, dtype=float)
    sNB = np.zeros_like(thr, dtype=float)

    if N == 0 or prevalence == 0.0:
        return NB, sNB, prevalence

    # 逐阈值计算 TP/FP
    for i, t in enumerate(thr):
        pred_pos = (y_prob >= t)
        TP = np.sum(pred_pos & (y_true == 1))
        FP = np.sum(pred_pos & (y_true == 0))
        nb = (TP / N) - (FP / N) * (t / (1.0 - t))
        NB[i] = nb
        sNB[i] = nb / prevalence

    return NB, sNB, prevalence


def decision_curve_treat_all_nb_and_snb(
    y_true: np.ndarray,
    thresholds: np.ndarray
):
    """
    Treat-all 策略：所有人都判为阳性（干预）。
    NB_all(t)  = prevalence - (1-prevalence)*t/(1-t)
    sNB_all(t) = NB_all(t)/prevalence
    """
    y_true = np.asarray(y_true).astype(int)
    thr = _validate_thresholds(thresholds)

    N = len(y_true)
    prevalence = float(np.mean(y_true)) if N > 0 else 0.0

    NB = np.zeros_like(thr, dtype=float)
    sNB = np.zeros_like(thr, dtype=float)

    if N == 0 or prevalence == 0.0:
        return NB, sNB, prevalence

    odds = thr / (1.0 - thr)
    NB = prevalence - (1.0 - prevalence) * odds
    sNB = NB / prevalence
    return NB, sNB, prevalence


def decision_curve_treat_none_nb_and_snb(
    y_true: np.ndarray,
    thresholds: np.ndarray
):
    """
    Treat-none：所有人都不干预 -> NB = 0, sNB = 0
    prevalence 仍返回，方便写入文件
    """
    y_true = np.asarray(y_true).astype(int)
    thr = _validate_thresholds(thresholds)
    N = len(y_true)
    prevalence = float(np.mean(y_true)) if N > 0 else 0.0
    NB = np.zeros_like(thr, dtype=float)
    sNB = np.zeros_like(thr, dtype=float)
    return NB, sNB, prevalence


# ==========================
# 主流程
# ==========================

def main():
    print(f"[INFO] 初始化 DCA 输出目录: {DCA_BASE_DIR}", flush=True)

    # 你原来是 0~0.05；保留
    thresholds = np.arange(0.0, 0.3000001, 0.0001)

    for outcome in OUTCOMES:
        print(f"\n[INFO] ===== 开始处理 outcome: {outcome} =====", flush=True)

        cv_data = {}
        test_data = {}

        # 只加载DCA需要的模型数据
        missing = False
        for model_tag in DCA_MODELS:
            cv_file = os.path.join(CV_DIR, f"{outcome}_{model_tag}_outer5fold_predictions.csv")
            test_file = os.path.join(TEST_DIR, f"{outcome}_{model_tag}_test_predictions.csv")

            if not os.path.exists(cv_file):
                print(f"[WARN] CV file not found for {outcome}, {model_tag}: {cv_file}", flush=True)
                missing = True
                break
            if not os.path.exists(test_file):
                print(f"[WARN] Test file not found for {outcome}, {model_tag}: {test_file}", flush=True)
                missing = True
                break

            print(f"[INFO] 读取文件: {cv_file}", flush=True)
            cv_data[model_tag] = pd.read_csv(cv_file)

            print(f"[INFO] 读取文件: {test_file}", flush=True)
            test_data[model_tag] = pd.read_csv(test_file)

        if missing:
            print(f"[WARN] outcome {outcome} 存在缺失文件，跳过处理。", flush=True)
            continue

        print(f"[INFO] outcome {outcome} 所有模型文件读取完成。", flush=True)

        # ==========================
        # CV DCA
        # ==========================
        print(f"[INFO] 计算 DCA 数据: outcome={outcome} (CV)...", flush=True)
        dca_cv_records = []

        y_cv = cv_data["model1_base"]["actual"].values.astype(int)

        # 各模型：NB + sNB
        for m in DCA_MODELS:
            probs_cv = cv_data[m]["pred_raw"].values.astype(float)
            NB_arr, sNB_arr, prev = decision_curve_nb_and_snb(y_cv, probs_cv, thresholds)

            for t, nb, snb in zip(thresholds, NB_arr, sNB_arr):
                dca_cv_records.append({
                    "outcome": outcome,
                    "model_type": m,
                    "thresholds": float(t),
                    "NB": float(nb),     # ✅ 未标准化
                    "sNB": float(snb),   # ✅ 标准化
                    "prevalence": float(prev)
                })

        # Treat-all / Treat-none：NB + sNB
        NB_all, sNB_all, prev_all = decision_curve_treat_all_nb_and_snb(y_cv, thresholds)
        NB_none, sNB_none, prev_none = decision_curve_treat_none_nb_and_snb(y_cv, thresholds)

        for t, nb, snb in zip(thresholds, NB_all, sNB_all):
            dca_cv_records.append({
                "outcome": outcome,
                "model_type": "treat_all",
                "thresholds": float(t),
                "NB": float(nb),
                "sNB": float(snb),
                "prevalence": float(prev_all)
            })

        for t, nb, snb in zip(thresholds, NB_none, sNB_none):
            dca_cv_records.append({
                "outcome": outcome,
                "model_type": "treat_none",
                "thresholds": float(t),
                "NB": float(nb),
                "sNB": float(snb),
                "prevalence": float(prev_none)
            })

        df_dca_cv = pd.DataFrame(dca_cv_records)
        dca_cv_file = os.path.join(DCA_CV_DIR, f"dca_{outcome}.csv")
        df_dca_cv.to_csv(dca_cv_file, index=False)
        print(f"[INFO] CV DCA 已保存: {dca_cv_file}", flush=True)

        # ==========================
        # Test DCA
        # ==========================
        print(f"[INFO] 计算 DCA 数据: outcome={outcome} (Test)...", flush=True)
        dca_test_records = []

        y_test = test_data["model1_base"]["actual"].values.astype(int)

        for m in DCA_MODELS:
            probs_test = test_data[m]["pred_raw"].values.astype(float)
            NB_arr, sNB_arr, prev = decision_curve_nb_and_snb(y_test, probs_test, thresholds)

            for t, nb, snb in zip(thresholds, NB_arr, sNB_arr):
                dca_test_records.append({
                    "outcome": outcome,
                    "model_type": m,
                    "thresholds": float(t),
                    "NB": float(nb),
                    "sNB": float(snb),
                    "prevalence": float(prev)
                })

        NB_all, sNB_all, prev_all = decision_curve_treat_all_nb_and_snb(y_test, thresholds)
        NB_none, sNB_none, prev_none = decision_curve_treat_none_nb_and_snb(y_test, thresholds)

        for t, nb, snb in zip(thresholds, NB_all, sNB_all):
            dca_test_records.append({
                "outcome": outcome,
                "model_type": "treat_all",
                "thresholds": float(t),
                "NB": float(nb),
                "sNB": float(snb),
                "prevalence": float(prev_all)
            })

        for t, nb, snb in zip(thresholds, NB_none, sNB_none):
            dca_test_records.append({
                "outcome": outcome,
                "model_type": "treat_none",
                "thresholds": float(t),
                "NB": float(nb),
                "sNB": float(snb),
                "prevalence": float(prev_none)
            })

        df_dca_test = pd.DataFrame(dca_test_records)
        dca_test_file = os.path.join(DCA_TEST_DIR, f"dca_{outcome}.csv")
        df_dca_test.to_csv(dca_test_file, index=False)
        print(f"[INFO] Test DCA 已保存: {dca_test_file}", flush=True)

        print(f"[INFO] ===== outcome {outcome} 处理完成 =====", flush=True)

    print(f"\n[INFO] 全部DCA计算完成，结果已保存至: {DCA_BASE_DIR}", flush=True)


if __name__ == "__main__":
    main()
