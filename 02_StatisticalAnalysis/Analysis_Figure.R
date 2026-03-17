library(tidyverse)
library(data.table)
library(ggplot2)
library(pROC)
library(patchwork)
library(ggpubr)
library(glue)
library(grid)
library(forcats)
library(stringr)
library(scales)
library(ggh4x)
library(pheatmap)
library(gtable)
library(ggforce)
library(arrow)



##################### ------ Figure2, epidemiology findings ------
dat_HR <- read_csv("Epi/dynamic_feature_HR.csv")

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggh4x)
  library(scales)
})


# 1) Outcomes (rows) and display names
outcome_label_map <- c(
  mi                       = "MI",
  afib_flutter             = "AF",
  stroke                   = "Stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  dementia                 = "Dementia",
  liver_cancer             = "Liver cancer"
)

outcomes_10 <- names(outcome_label_map)


# 2) Biomarkers (cols) and display names
biomarker_label_map <- c(
  BMI                 = "BMI",
  bpsystolic          = "SBP",
  Wbc                 = "WBC",
  fastingglucosemmol  = "Fasting glucose",
  AST                 = "AST",
  creatinine          = "Creatinine",
  LDL                 = "LDL cholesterol"
)

biomarkers_7 <- names(biomarker_label_map)


# 3) Map exposure -> biomarker + feature type
get_biomarker_from_exposure <- function(exposure) {
  # handle threshold proportion names
  if (str_detect(exposure, "^prop_high_")) {
    suffix <- str_remove(exposure, "^prop_high_")
    # map suffix to biomarker core used in exposures
    mp <- c(
      sbp        = "bpsystolic",
      glucose    = "fastingglucosemmol",
      LDL        = "LDL",
      creatinine = "creatinine",
      # dbp/TC/TG exist in exposures list but not in this 7-biomarker selection
      dbp = "bpdiastolic",
      TC  = "totalcholesterol",
      TG  = "triglycerides"
    )
    out <- unname(mp[suffix])
    return(out)
  }
  
  # remove prefixes to extract core
  core <- exposure %>%
    str_remove("^mean_") %>%
    str_remove("^baseline_") %>%
    str_remove("^sd_") %>%
    str_remove("^slope_") %>%
    str_remove("^annual_ratio_change_") %>%
    str_remove("^annual_log_change_")
  
  core
}

get_feature_type <- function(exposure) {
  case_when(
    str_detect(exposure, "^baseline_") ~ "Baseline level",
    str_detect(exposure, "^mean_")     ~ "Mean level",
    str_detect(exposure, "^sd_")       ~ "Variability",
    str_detect(exposure, "^slope_")    ~ "Slope (trend)",
    str_detect(exposure, "^annual_ratio_change_|^annual_log_change_") ~ "Annualized rate of change",
    str_detect(exposure, "^prop_high_") ~ "Threshold exceedance proportion",
    TRUE ~ NA_character_
  )
}

# Desired ordering of feature categories on x-axis
feature_levels <- c(
  "Baseline level",
  "Mean level",
  "Variability",
  "Slope (trend)",
  "Annualized rate of change",
  "Threshold exceedance proportion"
)


# 4) Build plotting dataset
plot_df <- dat_HR %>%
  filter(outcome_en %in% outcomes_10) %>%
  mutate(
    biomarker_core = map_chr(exposure, get_biomarker_from_exposure),
    feature_type   = get_feature_type(exposure)
  ) %>%
  filter(
    biomarker_core %in% biomarkers_7,
    !is.na(feature_type),
    is.finite(HR), is.finite(CI_low), is.finite(CI_high),
    HR > 0, CI_low > 0, CI_high > 0
  ) %>%
  mutate(
    outcome_label  = factor(outcome_label_map[outcome_en], levels = unname(outcome_label_map[outcomes_10])),
    biomarker_label = factor(biomarker_label_map[biomarker_core], levels = unname(biomarker_label_map[biomarkers_7])),
    feature_type   = factor(feature_type, levels = feature_levels)
  )


# 5) Colors
feat_cols <- c(
  "Baseline level"                  = "#B3B7BD",  
  "Mean level"                      = "#b1cbbb",
  "Variability"                     = "#80ced6", 
  "Slope (trend)"                   = "#4F7FB0", 
  "Annualized rate of change"       = "#034f84", 
  "Threshold exceedance proportion" = "#f18973"   
)


# 6) Plot
last_row_label <- tail(levels(plot_df$outcome_label), 1)

axisline_df <- plot_df %>%
  filter(outcome_label == last_row_label) %>%
  group_by(outcome_label, biomarker_label) %>%
  summarise(
    y_min = min(CI_low, na.rm = TRUE),   # 该 panel 的最小正值
    .groups = "drop"
  ) %>%
  mutate(
    # 往下挪一点点，但必须保持 >0（log轴要求）
    y_line = pmax(y_min * 0.97, 1e-6)
  )

p_hr_grid <- ggplot(plot_df, aes(x = feature_type, y = HR, color = feature_type)) +
  geom_hline(yintercept = 1, linetype = "dashed", linewidth = 0.35, color = "grey60") +
  geom_errorbar(aes(ymin = CI_low, ymax = CI_high), width = 0.18, linewidth = 0.40, alpha = 0.90) +
  geom_point(size = 1.35, alpha = 0.95) +
  scale_color_manual(values = feat_cols, drop = FALSE) +
  scale_y_log10(
    breaks = c(0.5, 0.7, 1, 1.5, 2, 3),
    labels = label_number(accuracy = 0.1)
  ) +
  labs(
    y = "HR",
    x = NULL
  ) + 
  geom_segment(
    data = axisline_df,
    aes(x = -Inf, xend = Inf, y = y_line, yend = y_line),
    inherit.aes = FALSE,
    linewidth = 0.35,
    color = "#2B2B2B"
  ) +
  ggh4x::facet_grid2(
    rows = vars(outcome_label),
    cols = vars(biomarker_label),
    scales = "free_y",
    independent = "y",
    axes = "all",
    switch = "y"
  ) +
  theme_classic(base_size = 11) +
  theme(
    # facet strips
    strip.background = element_blank(),
    strip.placement  = "outside",  # <<<<<< 加这一行：让行标题在panel外侧
    strip.text.x     = element_text(face = "bold", size = 10, color = "#1F2328"),
    strip.text.y.left= element_text(  # <<<<<< 用这个控制左侧行标题
      face = "bold", size = 10, color = "#1F2328",
      margin = margin(r = 6)
    ),
    strip.text.y     = element_blank(),
    
    # axes
    axis.title.x = element_blank(),
    axis.title.y = element_text(face = "bold", size = 11, margin = margin(r = 6)),
    axis.text.x  = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y  = element_text(size = 8, color = "#1F2328"),
    axis.ticks   = element_line(linewidth = 0.30, color = "#2B2B2B"),
    # axis.line    = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line.y  = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line.x  = element_blank(),
    
    # gridlines
    panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.25),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    
    # legend
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text  = element_text(size = 9),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    
    # spacing
    panel.spacing = unit(3.5, "pt"),
    plot.margin = margin(8, 10, 8, 8)
  ) +
  guides(
    color = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(size = 2.2))
  )

p_hr_grid

ggsave("Fig2_HR.pdf", p_hr_grid, width = 13, height = 11)











##################### ------ Figure3, model performance ------
###### auroc绘图
dat_res <- read_csv("XGBoost/summary_metrics.csv")

# 1) outcome 显示名
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)
panel_levels <- unname(outcome_label_map)

# 2) 模型顺序与显示名（model1 -> model5）
model_order <- c(
  "model1_base",
  "model2_clinical",
  "model3_dynamic",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_label_map <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

# 3) 低饱和度配色
model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3",
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)

# 4) p值 -> 显著性标签
p_to_sig <- function(p) {
  ifelse(
    is.na(p), "",
    ifelse(p < 0.001, "***",
           ifelse(p < 0.01, "**",
                  ifelse(p < 0.05, "*", "ns")))
  )
}

# 5) 整理数据 + 计算标注高度
plot_dat <- dat_res %>%
  filter(model_type %in% model_order) %>%
  mutate(
    outcome_label = recode(outcome, !!!outcome_label_map),
    outcome_label = factor(outcome_label, levels = panel_levels), 
    model_type    = factor(model_type, levels = model_order),
    model_label   = factor(recode(as.character(model_type), !!!model_label_map),
                           levels = model_label_map[model_order]),
    x_num         = as.integer(model_type),
    sig_label     = ifelse(as.character(model_type) == "model5_clinical_dynamic_exam",
                           "ref",
                           p_to_sig(cv_vs_model5_auroc_pvalue))
  ) %>%
  select(
    outcome_label, model_label, x_num,
    cv_roc_auc_mean, cv_roc_auc_ci_lower, cv_roc_auc_ci_upper,
    sig_label
  ) %>%
  group_by(outcome_label) %>%
  mutate(
    y_span = max(cv_roc_auc_ci_upper, na.rm = TRUE) - min(cv_roc_auc_ci_lower, na.rm = TRUE),
    y_off  = ifelse(is.finite(y_span) & y_span > 0, 0.07 * y_span, 0.006),
    y_sig  = cv_roc_auc_ci_upper + y_off
  ) %>%
  ungroup()

# 6) 作图
p <- ggplot(plot_dat, aes(x = x_num, y = cv_roc_auc_mean)) +
  geom_line(aes(group = 1), linewidth = 0.55, color = "#C7CDD1") +
  geom_linerange(
    aes(ymin = cv_roc_auc_ci_lower, ymax = cv_roc_auc_ci_upper),
    linewidth = 0.55,
    color = "#3F4852"
  ) +
  geom_point(aes(color = model_label), size = 2.4) +
  geom_text(
    aes(y = y_sig, label = sig_label),
    size = 3.3,
    color = "#2F343A",
    vjust = 0
  ) +
  scale_color_manual(values = model_cols, name = NULL) +
  scale_x_continuous(
    breaks = 1:5,
    labels = rep("", 5),
    expand = expansion(mult = c(0.06, 0.06))
  ) +
  scale_y_continuous(
    breaks = pretty_breaks(n = 4),
    labels = label_number(accuracy = 0.001),
    expand = expansion(mult = c(0.06, 0.18))
  ) +
  labs(
    title = "Nested Cross-Validation AUROC",
    x = NULL, y = NULL
  ) +
  ggh4x::facet_wrap2(
    ~ outcome_label,
    nrow = 4, ncol = 5,
    scales = "free_y",
    axes = "all"
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14, color = "#1F2328", hjust = 0.5),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10, color = "#1F2328"),
    
    axis.text.x = element_blank(),
    axis.ticks.x = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.text.y = element_text(size = 9, color = "#1F2328"),
    axis.ticks.y = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line = element_line(linewidth = 0.35, color = "#2B2B2B"),
    
    panel.grid.major.y = element_line(
      color = "#F4F6F8",
      linewidth = 0.3
    ),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.text = element_text(size = 9.5, color = "#1F2328"),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    
    plot.margin = margin(10, 12, 8, 10)
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(size = 3))) +
  coord_cartesian(clip = "off")

p

ggsave("Fig3_AUROC.pdf", p, width = 10, height = 7)




###### 校准曲线
# 0) Paths
base_dir   <- "XGBoost"
pred_dir   <- file.path(base_dir, "outer_cv_predictions")

# 1) Outcomes 
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)
outcomes_20  <- names(outcome_label_map)
panel_levels <- unname(outcome_label_map)

# 2) Models (5) 
model_types_5 <- c(
  "model1_base",
  "model2_clinical",
  "model3_dynamic",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

# 3) Colors
model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3",
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)

# 4) Read prediction file
read_pred_one <- function(outcome, model_type) {
  f <- file.path(pred_dir, sprintf("%s_%s_outer5fold_predictions.csv", outcome, model_type))
  if (!file.exists(f)) {
    warning("Missing prediction file: ", f)
    return(NULL)
  }
  readr::read_csv(f, show_col_types = FALSE) %>%
    transmute(
      outcome = outcome,
      model_type = model_type,
      actual = as.numeric(actual),
      pred   = as.numeric(pred_raw)
    ) %>%
    filter(!is.na(actual), !is.na(pred)) %>%
    mutate(pred = pmin(pmax(pred, 0), 1))
}

pred_df <- bind_rows(lapply(outcomes_20, function(oc) {
  bind_rows(lapply(model_types_5, function(mt) read_pred_one(oc, mt)))
}))

# 5) Quantile-binned calibration lines
get_calib_bins <- function(df_one, n_bins = 20) {
  if (nrow(df_one) < n_bins) return(NULL)
  
  df_one %>%
    mutate(bin = ntile(pred, n_bins)) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(pred) * 100,
      obs_rate  = mean(actual) * 100,
      .groups = "drop"
    ) %>%
    arrange(mean_pred)
}

n_bins <- 20

calib_df <- pred_df %>%
  group_by(outcome, model_type) %>%
  group_modify(~{
    out <- get_calib_bins(.x, n_bins = n_bins)
    if (is.null(out)) return(tibble())
    out
  }) %>%
  ungroup() %>%
  mutate(
    outcome_label = factor(outcome_label_map[outcome], levels = panel_levels),
    model = factor(model_type_labels[model_type],
                   levels = unname(model_type_labels[model_types_5]))
  )

# 6) Better per-panel axis maxima
q_focus <- 0.99          
min_cap <- 0.25          
max_cap <- 100

panel_limits <- calib_df %>%
  group_by(outcome_label) %>%
  summarise(
    x_q = suppressWarnings(quantile(mean_pred, probs = q_focus, na.rm = TRUE)),
    y_q = suppressWarnings(quantile(obs_rate,  probs = q_focus, na.rm = TRUE)),
    x_max = max(mean_pred, na.rm = TRUE),
    y_max = max(obs_rate,  na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    m_q = pmax(ifelse(is.finite(x_q), x_q, x_max),
               ifelse(is.finite(y_q), y_q, y_max),
               na.rm = TRUE),
    
    m_max = m_q + pmax(0.15 * m_q, 0.05),
    
    m_max = pmin(pmax(m_max, min_cap), max_cap)
  )

# 7) Panel-specific breaks/labels
make_breaks <- function(maxv) {
  # Choose step based on maxv
  step <- if (maxv <= 0.5) 0.1 else if (maxv <= 1) 0.2 else if (maxv <= 2) 0.5 else if (maxv <= 10) 2 else if (maxv <= 25) 5 else 10
  seq(0, maxv, by = step)
}

make_labels <- function(maxv) {
  # show one decimal for very small ranges
  if (maxv <= 1) label_number(accuracy = 0.1) else label_number(accuracy = 1)
}

x_scales <- lapply(panel_levels, function(lbl) {
  maxv <- panel_limits$m_max[match(lbl, panel_limits$outcome_label)]
  scale_x_continuous(
    limits = c(0, maxv),
    breaks = make_breaks(maxv),
    labels = make_labels(maxv),
    expand = expansion(mult = c(0.02, 0.04))
  )
})

y_scales <- lapply(panel_levels, function(lbl) {
  maxv <- panel_limits$m_max[match(lbl, panel_limits$outcome_label)]
  scale_y_continuous(
    limits = c(0, maxv),
    breaks = make_breaks(maxv),
    labels = make_labels(maxv),
    expand = expansion(mult = c(0.02, 0.04))
  )
})

# 8) Theme
theme_calib <- theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14, color = "#1F2328", hjust = 0.5),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10, color = "#1F2328"),
    
    axis.title.x = element_text(
      size = 11,
      face = "bold",        
      color = "#1F2328",
      margin = margin(t = 6)
    ),
    axis.title.y = element_text(
      size = 11,
      face = "bold",       
      color = "#1F2328",
      margin = margin(r = 6)
    ),
    axis.text = element_text(size = 9, color = "#1F2328"),
    axis.ticks = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line  = element_line(linewidth = 0.35, color = "#2B2B2B"),
    
    panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
    panel.grid.major.x = element_line(color = "#F5F7FA", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 9.5, color = "#1F2328"),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    
    plot.margin = margin(10, 12, 8, 10)
  )

# 9) Plot
p_calib <- ggplot(calib_df, aes(x = mean_pred, y = obs_rate, color = model, group = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.35, color = "#8B949E") +
  geom_line(linewidth = 0.85, alpha = 0.95) +
  scale_color_manual(values = model_cols) +
  labs(
    title = "Calibration Curve",
    x = "Predicted risk (%)",
    y = "Observed event rate (%)"
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_calib +
  ggh4x::facet_wrap2(~ outcome_label, nrow = 4, ncol = 5, scales = "free", axes = "all") +
  ggh4x::facetted_pos_scales(x = x_scales, y = y_scales)

p_calib

ggsave("Fig3_calibration.pdf", p_calib, width = 10, height = 7)











##################### ------ Figure4, clinical benefit ------
###### DCA
suppressPackageStartupMessages({
  library(tidyverse)
  library(glue)
  library(scales)
})

# 0) Paths
base_dir   <- "XGBoost"
dca_dir    <- file.path(base_dir, "dca_outputs", "cv")

# 1) Outcomes (12) + labels
outcome_label_map_12 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease"
)
outcomes_12  <- names(outcome_label_map_12)
panel_levels <- unname(outcome_label_map_12)

# 2) Models (4) + labels
model_types_4 <- c(
  "model1_base",
  "model2_clinical",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)
model_levels_4 <- unname(model_type_labels[model_types_4])

# 3) Colors
model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3", # not used
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)

fill_color <- scales::alpha(model_cols[["Model 5: Clin+Exam (longitudinal)"]], 0.25)

# 4) Read DCA CSVs
read_dca_one <- function(outcome) {
  f <- file.path(dca_dir, glue("dca_{outcome}.csv"))
  if (!file.exists(f)) {
    warning("Missing DCA file: ", f)
    return(NULL)
  }
  
  readr::read_csv(f, show_col_types = FALSE) %>%
    filter(model_type %in% c(model_types_4, "treat_all", "treat_none")) %>%
    transmute(
      outcome       = outcome,
      model_type    = as.character(model_type),
      thresholds    = as.numeric(thresholds) * 100,  # -> %
      NB            = as.numeric(NB),
      sNB           = as.numeric(sNB) * 100,          # -> %
      prevalence    = as.numeric(prevalence),
      outcome_label = factor(outcome_label_map_12[outcome], levels = panel_levels)
    ) %>%
    filter(!is.na(thresholds), !is.na(sNB))
}

dca_df_raw <- bind_rows(lapply(outcomes_12, read_dca_one))

# 5) Keep model and reference data, add labels
dca_models_raw <- dca_df_raw %>%
  filter(model_type %in% model_types_4) %>%
  mutate(
    model = model_type_labels[model_type],
    model = factor(model, levels = model_levels_4)
  )

dca_ref <- dca_df_raw %>%
  filter(model_type %in% c("treat_all", "treat_none")) %>%
  group_by(outcome_label, model_type) %>%
  arrange(thresholds) %>%
  ungroup()

# 6) Global threshold truncation
model5_type <- "model5_clinical_dynamic_exam"

thr_limits <- dca_models_raw %>%
  filter(model_type == model5_type) %>%
  group_by(outcome_label) %>%
  arrange(thresholds) %>%
  summarise(
    thr_cut = suppressWarnings(min(thresholds[sNB <= 0], na.rm = TRUE)),
    thr_max_data = suppressWarnings(max(thresholds, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(
    thr_cut = ifelse(is.finite(thr_cut), thr_cut, thr_max_data),
    thr_max = pmin(thr_cut + 2, thr_max_data),
    thr_max = pmax(thr_max, 5)
  )

dca_models_raw <- dca_models_raw %>%
  left_join(thr_limits, by = "outcome_label") %>%
  filter(thresholds <= thr_max) %>%
  select(-thr_cut, -thr_max_data, -thr_max)

dca_ref <- dca_ref %>%
  left_join(thr_limits, by = "outcome_label") %>%
  filter(thresholds <= thr_max) %>%
  select(-thr_cut, -thr_max_data, -thr_max)

# 7) Enforce "floor at 0" and "absorbing at 0" for model curves
# Rule:
#   - any sNB < 0 -> 0
#   - once sNB reaches 0 (after flooring), keep sNB=0 for all later thresholds
enforce_floor_absorb0 <- function(df) {
  df <- df %>% arrange(thresholds)
  s <- pmax(df$sNB, 0)
  first_zero <- which(s <= 0)[1]
  if (!is.na(first_zero)) {
    s[first_zero:length(s)] <- 0
  }
  df$sNB <- s
  df
}

dca_models <- dca_models_raw %>%
  group_by(outcome_label, model_type) %>%
  group_modify(~ enforce_floor_absorb0(.x)) %>%
  ungroup() %>%
  mutate(
    # keep ordering stable
    model = factor(model_type_labels[model_type], levels = model_levels_4)
  )

# 8) Ribbon between Model 4 and Model 5 (use adjusted sNB)
dca_ribbon <- dca_models %>%
  filter(model %in% c("Model 4: Clin+Exam (baseline)",
                      "Model 5: Clin+Exam (longitudinal)")) %>%
  select(outcome_label, thresholds, model, sNB) %>%
  pivot_wider(names_from = model, values_from = sNB) %>%
  drop_na(`Model 4: Clin+Exam (baseline)`, `Model 5: Clin+Exam (longitudinal)`) %>%
  arrange(outcome_label, thresholds) %>%
  mutate(
    ymin = pmin(`Model 4: Clin+Exam (baseline)`, `Model 5: Clin+Exam (longitudinal)`),
    ymax = pmax(`Model 4: Clin+Exam (baseline)`, `Model 5: Clin+Exam (longitudinal)`)
  )

# 9) Theme
theme_dca <- theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14, color = "#1F2328", hjust = 0.5),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10, color = "#1F2328"),
    
    axis.title.x = element_text(size = 11, face = "bold", color = "#1F2328", margin = margin(t = 6)),
    axis.title.y = element_text(size = 11, face = "bold", color = "#1F2328", margin = margin(r = 6)),
    axis.text = element_text(size = 9, color = "#1F2328"),
    
    axis.ticks = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line  = element_line(linewidth = 0.45, color = "#2B2B2B"),
    
    panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
    panel.grid.major.x = element_line(color = "#F5F7FA", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 9.5, color = "#1F2328"),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    
    plot.margin = margin(10, 12, 8, 10)
  )

# 10) Plot
treat_all_df  <- dca_ref %>% filter(model_type == "treat_all")
treat_none_df <- dca_ref %>% filter(model_type == "treat_none")

p_dca <- ggplot() +
  # Treat-none
  geom_line(
    data = treat_none_df,
    aes(x = thresholds, y = sNB, group = outcome_label),
    color = "#6B7280",
    linewidth = 0.45,
    linetype = "solid",
    alpha = 0.95
  ) +
  # Treat-all
  geom_line(
    data = treat_all_df,
    aes(x = thresholds, y = sNB, group = outcome_label),
    color = "#9CA3AF",
    linewidth = 0.45,
    linetype = "solid",
    alpha = 0.95
  ) +
  # Ribbon between Model 4 and Model 5
  geom_ribbon(
    data = dca_ribbon,
    aes(x = thresholds, ymin = ymin, ymax = ymax, group = outcome_label),
    fill = fill_color
  ) +
  # Model curves (ALL SOLID)
  geom_line(
    data = dca_models,
    aes(x = thresholds, y = sNB, color = model, group = interaction(outcome_label, model)),
    linewidth = 0.7,
    alpha = 0.95,
    linetype = "solid"
  ) +
  facet_wrap(~ outcome_label, nrow = 3, ncol = 4, scales = "free_x") +
  scale_color_manual(values = model_cols[model_levels_4]) +
  coord_cartesian(ylim = c(-1, NA)) +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.04))) +
  labs(
    x = "Threshold probability (%)",
    y = "Standardized net benefit (%)"
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_dca

p_dca

ggsave("Fig4_DCA.pdf", p_dca, width = 10, height = 7)






###### DR 曲线
dat_DR <- read_csv("XGBoost/DR_curve/dr_curve_fpr_grid.csv")

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(scales)
})


# 1) Outcomes (12) + titles (standard)
outcome_label_map_12 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease"
)

outcomes_12 <- names(outcome_label_map_12)


# 2) Models (3) + labels + colors
model_types_3 <- c(
  "model2_clinical",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

model_levels_3 <- unname(model_type_labels[model_types_3])

model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3",
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)


# 3) Prepare data: convert to % and standardize factors
dr_df <- dat_DR %>%
  filter(outcome %in% outcomes_12, model %in% model_types_3) %>%
  transmute(
    outcome,
    outcome_label = factor(outcome_label_map_12[outcome], levels = unname(outcome_label_map_12)),
    model_type = model,
    model = factor(model_type_labels[model_type], levels = model_levels_3),
    FDR = as.numeric(FPR_grid) * 100,  # user wants label "FDR (%)"
    DRp = as.numeric(DR) * 100
  ) %>%
  filter(!is.na(FDR), !is.na(DRp)) %>%
  group_by(outcome, model) %>%
  arrange(FDR, .by_group = TRUE) %>%
  ungroup()


# 4) Helper: per-outcome y-axis max (dynamic, padded)
y_limits <- dr_df %>%
  group_by(outcome) %>%
  summarise(
    y_max_raw = max(DRp, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    # Add padding; keep >= 5% headroom; handle near-zero
    y_max = pmax(5, y_max_raw * 1.10),
    y_max = ifelse(is.finite(y_max), y_max, 5)
  )

get_ymax <- function(outcome_key) {
  y_limits$y_max[match(outcome_key, y_limits$outcome)]
}

# Nice y breaks
nice_breaks <- function(maxv) {
  if (maxv <= 5)   return(seq(0, maxv, by = 1))
  if (maxv <= 10)  return(seq(0, maxv, by = 2))
  if (maxv <= 25)  return(seq(0, maxv, by = 5))
  if (maxv <= 50)  return(seq(0, maxv, by = 10))
  if (maxv <= 100) return(seq(0, maxv, by = 25))
  seq(0, maxv, by = 50)
}


# 5) Theme
theme_dr <- function() {
  theme_classic(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
      axis.title.x = element_text(face = "bold", size = 11, margin = margin(t = 6)),
      axis.title.y = element_text(face = "bold", size = 11, margin = margin(r = 6)),
      axis.text = element_text(size = 9, color = "#1F2328"),
      axis.ticks = element_line(linewidth = 0.35, color = "#2B2B2B"),
      axis.line  = element_line(linewidth = 0.45, color = "#2B2B2B"),
      panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
      panel.grid.major.x = element_line(color = "#F5F7FA", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 10),
      legend.key.width = unit(18, "pt"),
      legend.spacing.x = unit(10, "pt"),
      plot.margin = margin(6, 6, 6, 6)
    )
}


# 6) Build one panel plot
make_dr_panel <- function(outcome_key, show_x = FALSE, show_y = FALSE) {
  df_one <- dr_df %>% filter(outcome == outcome_key)
  ymax  <- get_ymax(outcome_key)
  
  ggplot(df_one, aes(x = FDR, y = DRp, color = model, group = model)) +
    geom_line(linewidth = 0.95, linetype = "solid", alpha = 0.95) +
    geom_point(size = 1.3, alpha = 0.85) +
    scale_color_manual(values = model_cols[model_levels_3]) +
    scale_x_continuous(
      limits = c(0, 41),
      breaks = c(0, 10, 20, 30, 40),
      expand = expansion(mult = c(0.01, 0.02))
    ) +
    scale_y_continuous(
      limits = c(0, ymax),
      breaks = nice_breaks(ymax),
      labels = label_number(accuracy = 1),
      expand = expansion(mult = c(0.01, 0.03))
    ) +
    labs(
      title = outcome_label_map_12[[outcome_key]],
      x = if (show_x) "FDR (%)" else NULL,
      y = if (show_y) "DR (%)" else NULL
    ) +
    theme_dr() +
    theme(
      # If not showing axis titles, also remove extra margin so panels align nicely
      axis.title.x = if (show_x) element_text() else element_blank(),
      axis.title.y = if (show_y) element_text() else element_blank()
    )
}


# 7) 3x4 layout order
outcomes_order <- outcomes_12 

panels <- list()
for (i in seq_along(outcomes_order)) {
  oc <- outcomes_order[i]
  row <- ceiling(i / 4)
  col <- i - (row - 1) * 4
  show_x <- (row == 3)          # bottom row
  show_y <- (col == 1)          # left column
  panels[[i]] <- make_dr_panel(oc, show_x = show_x, show_y = show_y)
}

# Combine and collect legend at bottom
p_dr <- wrap_plots(panels, nrow = 3, ncol = 4) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

p_dr

ggsave("Fig4_DR.pdf", p_dr, width = 10, height = 7)













##################### ------ Figure5, external validation ------
###### 地理外部验证
dat_geo_val <- read_csv("XGBoost/summary_metrics.csv")

### model2 vs model5
suppressPackageStartupMessages({
  library(tidyverse)
  library(scales)
})

# 1) Disease labels
outcome_label_map_12 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease"
)

outcomes_12 <- names(outcome_label_map_12)


# 2) Model labels & colors
model_labels <- c(
  model2_clinical              = "Model 2: Clinical",
  model5_clinical_dynamic_exam = "Model 5: Clin+Exam (longitudinal)"
)

model_cols <- c(
  "Model 2: Clinical"                 = "#abd9e9",
  "Model 5: Clin+Exam (longitudinal)" = "#fc9272"
)


# 3) Prepare AUROC data
auc_df <- dat_geo_val %>%
  filter(
    outcome %in% outcomes_12,
    model_type %in% names(model_labels)
  ) %>%
  transmute(
    outcome,
    outcome_label = factor(
      outcome_label_map_12[outcome],
      levels = unname(outcome_label_map_12)
    ),
    model = model_labels[model_type],
    auc   = test_roc_auc,
    p_val = test_vs_model5_auroc_pvalue
  )


# 4) One row per outcome
plot_df <- auc_df %>%
  select(outcome, outcome_label, model, auc) %>%
  pivot_wider(
    names_from  = model,
    values_from = auc
  ) %>%
  left_join(
    auc_df %>%
      filter(model == "Model 2: Clinical") %>%
      select(outcome, p_val),
    by = "outcome"
  )


# 5) Significance labels
plot_df <- plot_df %>%
  mutate(
    sig = case_when(
      is.na(p_val)     ~ "",
      p_val < 0.001    ~ "***",
      p_val < 0.01     ~ "**",
      p_val < 0.05     ~ "*",
      TRUE             ~ "ns"
    )
  )


# 6) Y-axis limits
y_min <- min(plot_df$`Model 2: Clinical`,
             plot_df$`Model 5: Clin+Exam (longitudinal)`,
             na.rm = TRUE)

y_max <- max(plot_df$`Model 2: Clinical`,
             plot_df$`Model 5: Clin+Exam (longitudinal)`,
             na.rm = TRUE)

pad <- (y_max - y_min) * 0.03


# 7) Dumbbell plot
p_geo_auc <- ggplot(plot_df, aes(x = outcome_label)) +
  
  # connecting line
  geom_segment(
    aes(
      xend = outcome_label,
      y    = `Model 2: Clinical`,
      yend = `Model 5: Clin+Exam (longitudinal)`
    ),
    color = "grey70",
    linewidth = 0.9
  ) +
  
  # Model 2 point
  geom_point(
    aes(y = `Model 2: Clinical`, color = "Model 2: Clinical"),
    size = 2.8
  ) +
  
  # Model 5 point
  geom_point(
    aes(y = `Model 5: Clin+Exam (longitudinal)`,
        color = "Model 5: Clin+Exam (longitudinal)"),
    size = 2.8
  ) +
  
  # significance
  geom_text(
    aes(
      y = pmax(`Model 2: Clinical`,
               `Model 5: Clin+Exam (longitudinal)`) + 0.005,
      label = sig
    ),
    size = 3.2
  ) + 
  
  scale_color_manual(values = model_cols) +
  
  scale_y_continuous(
    limits = c(y_min - pad, y_max + pad),
    breaks = pretty_breaks(5),
    labels = label_number(accuracy = 0.01)
  ) +
  
  labs(
    x = NULL,
    y = "AUROC"
  ) +
  
  theme_classic(base_size = 11) +
  theme(
    axis.text.x = element_text(
      angle = 90, vjust = 0.5, hjust = 1, size = 9
    ),
    axis.title.y = element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 9),
    plot.margin = margin(8, 10, 8, 8)
  )

p_geo_auc

ggsave("Fig5_geo_model2_vs_model5_auroc.pdf", p_geo_auc, width = 6, height = 6)


### model4 vs model5
suppressPackageStartupMessages({
  library(tidyverse)
  library(scales)
})

# 1) Disease labels
outcome_label_map_12 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease"
)

outcomes_12 <- names(outcome_label_map_12)


# 2) Model labels & colors
model_labels <- c(
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

model_cols <- c(
  "Model 4: Clin+Exam (baseline)"     = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)" = "#fc9272"
)


# 3) Prepare AUROC data
auc_df <- dat_geo_val %>%
  filter(
    outcome %in% outcomes_12,
    model_type %in% names(model_labels)
  ) %>%
  transmute(
    outcome,
    outcome_label = factor(
      outcome_label_map_12[outcome],
      levels = unname(outcome_label_map_12)
    ),
    model = model_labels[model_type],
    auc   = test_roc_auc,
    p_val = test_vs_model5_auroc_pvalue
  )


# 4) One row per outcome
plot_df <- auc_df %>%
  select(outcome, outcome_label, model, auc) %>%
  pivot_wider(
    names_from  = model,
    values_from = auc
  ) %>%
  left_join(
    auc_df %>%
      filter(model == "Model 4: Clin+Exam (baseline)") %>%
      select(outcome, p_val),
    by = "outcome"
  )


# 5) Significance labels
plot_df <- plot_df %>%
  mutate(
    sig = case_when(
      is.na(p_val)     ~ "",
      p_val < 0.001    ~ "***",
      p_val < 0.01     ~ "**",
      p_val < 0.05     ~ "*",
      TRUE             ~ "ns"
    )
  )


# 6) Y-axis limits
y_min <- min(plot_df$`Model 4: Clin+Exam (baseline)`,
             plot_df$`Model 5: Clin+Exam (longitudinal)`,
             na.rm = TRUE)

y_max <- max(plot_df$`Model 4: Clin+Exam (baseline)`,
             plot_df$`Model 5: Clin+Exam (longitudinal)`,
             na.rm = TRUE)

pad <- (y_max - y_min) * 0.03


# 7) Dumbbell plot
p_geo_auc <- ggplot(plot_df, aes(x = outcome_label)) +
  
  # connecting line
  geom_segment(
    aes(
      xend = outcome_label,
      y    = `Model 4: Clin+Exam (baseline)`,
      yend = `Model 5: Clin+Exam (longitudinal)`
    ),
    color = "grey70",
    linewidth = 0.9
  ) +
  
  # Model 2 point
  geom_point(
    aes(y = `Model 4: Clin+Exam (baseline)`,
        color = "Model 4: Clin+Exam (baseline)"),
    size = 2.8
  ) +
  
  # Model 5 point
  geom_point(
    aes(y = `Model 5: Clin+Exam (longitudinal)`,
        color = "Model 5: Clin+Exam (longitudinal)"),
    size = 2.8
  ) +
  
  # significance
  geom_text(
    aes(
      y = pmax(`Model 4: Clin+Exam (baseline)`,
               `Model 5: Clin+Exam (longitudinal)`) + 0.005,
      label = sig
    ),
    size = 3.2
  ) + 
  
  scale_color_manual(values = model_cols) +
  
  scale_y_continuous(
    limits = c(y_min - pad, y_max + pad),
    breaks = pretty_breaks(5),
    labels = label_number(accuracy = 0.01)
  ) +
  
  labs(
    x = NULL,
    y = "AUROC"
  ) +
  
  theme_classic(base_size = 11) +
  theme(
    axis.text.x = element_text(
      angle = 90, vjust = 0.5, hjust = 1, size = 9
    ),
    axis.title.y = element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 9),
    plot.margin = margin(8, 10, 8, 8)
  )

p_geo_auc

ggsave("Fig5_geo_model4_vs_model5_auroc.pdf", p_geo_auc, width = 6, height = 6)


###### 地理外部验证，校准曲线
suppressPackageStartupMessages({
  library(tidyverse)
  library(scales)
  library(ggh4x)
})


# 0) Paths
pred_dir <- "XGBoost/external_val_predictions"


# 1) Outcomes
outcome_label_map_12 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease"
)

outcomes_12   <- names(outcome_label_map_12)
panel_levels <- unname(outcome_label_map_12)


# 2) Models
model_types <- c(
  "model2_clinical",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_type_labels <- c(
  model2_clinical               = "Model 2: Clinical",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

model_cols <- c(
  "Model 2: Clinical"                 = "#abd9e9",
  "Model 4: Clin+Exam (baseline)"     = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)" = "#fc9272"
)


# 3) Read prediction files
read_pred_one <- function(outcome, model_type) {
  f <- file.path(
    pred_dir,
    sprintf("%s_%s_test_predictions.csv", outcome, model_type)
  )
  
  if (!file.exists(f)) {
    warning("Missing file: ", f)
    return(NULL)
  }
  
  readr::read_csv(f, show_col_types = FALSE) %>%
    transmute(
      outcome    = outcome,
      model_type = model_type,
      actual     = as.numeric(actual),
      pred       = as.numeric(pred_raw)
    ) %>%
    filter(!is.na(actual), !is.na(pred)) %>%
    mutate(pred = pmin(pmax(pred, 0), 1))
}

pred_df <- bind_rows(lapply(outcomes_12, function(oc) {
  bind_rows(lapply(model_types, function(mt) {
    read_pred_one(oc, mt)
  }))
}))


# 4) Quantile-binned calibration
get_calib_bins <- function(df_one, n_bins = 20) {
  if (nrow(df_one) < n_bins) return(NULL)
  
  df_one %>%
    mutate(bin = ntile(pred, n_bins)) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(pred) * 100,
      obs_rate  = mean(actual) * 100,
      .groups = "drop"
    ) %>%
    arrange(mean_pred)
}

n_bins <- 20

calib_df <- pred_df %>%
  group_by(outcome, model_type) %>%
  group_modify(~{
    out <- get_calib_bins(.x, n_bins = n_bins)
    if (is.null(out)) return(tibble())
    out
  }) %>%
  ungroup() %>%
  mutate(
    outcome_label = factor(outcome_label_map_12[outcome],
                           levels = panel_levels),
    model = factor(
      model_type_labels[model_type],
      levels = unname(model_type_labels[model_types])
    )
  )


# 5) Per-panel axis limits
q_focus <- 0.99
min_cap <- 0.25
max_cap <- 100

panel_limits <- calib_df %>%
  group_by(outcome_label) %>%
  summarise(
    x_q = quantile(mean_pred, probs = q_focus, na.rm = TRUE),
    y_q = quantile(obs_rate,  probs = q_focus, na.rm = TRUE),
    x_max = max(mean_pred, na.rm = TRUE),
    y_max = max(obs_rate,  na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    m_q   = pmax(x_q, y_q, na.rm = TRUE),
    m_max = pmin(pmax(m_q + pmax(0.15 * m_q, 0.05), min_cap), max_cap)
  )

make_breaks <- function(maxv) {
  step <- if (maxv <= 0.5) 0.1 else if (maxv <= 1) 0.2 else if (maxv <= 2) 0.5 else if (maxv <= 10) 2 else if (maxv <= 25) 5 else 10
  seq(0, maxv, by = step)
}

make_labels <- function(maxv) {
  if (maxv <= 1) label_number(accuracy = 0.1)
  else label_number(accuracy = 1)
}

x_scales <- lapply(panel_levels, function(lbl) {
  maxv <- panel_limits$m_max[match(lbl, panel_limits$outcome_label)]
  scale_x_continuous(
    limits = c(0, maxv),
    breaks = make_breaks(maxv),
    labels = make_labels(maxv),
    expand = expansion(mult = c(0.02, 0.04))
  )
})

y_scales <- lapply(panel_levels, function(lbl) {
  maxv <- panel_limits$m_max[match(lbl, panel_limits$outcome_label)]
  scale_y_continuous(
    limits = c(0, maxv),
    breaks = make_breaks(maxv),
    labels = make_labels(maxv),
    expand = expansion(mult = c(0.02, 0.04))
  )
})


# 6) Theme
theme_calib <- theme_classic(base_size = 11) +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10),
    
    axis.title.x = element_text(face = "bold", size = 11, margin = margin(t = 6)),
    axis.title.y = element_text(face = "bold", size = 11, margin = margin(r = 6)),
    axis.text = element_text(size = 9),
    
    axis.ticks = element_line(linewidth = 0.35),
    axis.line  = element_line(linewidth = 0.35),
    
    panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
    panel.grid.major.x = element_line(color = "#F5F7FA", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 9.5),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    
    plot.margin = margin(8, 10, 8, 8)
  )


# 7) Plot
p_calib_ext <- ggplot(
  calib_df,
  aes(x = mean_pred, y = obs_rate, color = model, group = model)
) +
  geom_abline(
    slope = 1, intercept = 0,
    linetype = "dashed",
    linewidth = 0.35,
    color = "#8B949E"
  ) +
  geom_line(linewidth = 0.9, alpha = 0.95) +
  scale_color_manual(values = model_cols) +
  labs(
    x = "Predicted risk (%)",
    y = "Observed event rate (%)"
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_calib +
  ggh4x::facet_wrap2(
    ~ outcome_label,
    nrow = 3,
    ncol = 4,
    scales = "free",
    axes = "all"
  ) +
  ggh4x::facetted_pos_scales(
    x = x_scales,
    y = y_scales
  )

p_calib_ext

ggsave("Fig5_geo_calib.pdf", p_calib_ext, height = 6, width = 9)



###### geo dca
suppressPackageStartupMessages({
  library(tidyverse)
  library(glue)
  library(scales)
})

# 0) Paths
base_dir <- "XGBoost"
dca_dir  <- file.path(base_dir, "dca_outputs", "test") 

# 1) Outcomes (12) + labels
outcome_label_map_12 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease"
)
outcomes_12  <- names(outcome_label_map_12)
panel_levels <- unname(outcome_label_map_12)

# 2) Models (ONLY 3) + labels
model_types_3 <- c(
  "model2_clinical",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

model_levels_3 <- unname(model_type_labels[model_types_3])

# 3) Colors 
model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3",
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)

fill_color <- scales::alpha(model_cols[["Model 5: Clin+Exam (longitudinal)"]], 0.25)

# 4) Read DCA CSVs
read_dca_one <- function(outcome) {
  f <- file.path(dca_dir, glue("dca_{outcome}.csv"))
  if (!file.exists(f)) {
    warning("Missing DCA file: ", f)
    return(NULL)
  }
  
  readr::read_csv(f, show_col_types = FALSE) %>%
    filter(model_type %in% c(model_types_3, "treat_all", "treat_none")) %>%
    transmute(
      outcome       = outcome,
      model_type    = as.character(model_type),
      thresholds    = as.numeric(thresholds) * 100,   # -> %
      NB            = as.numeric(NB),
      sNB           = as.numeric(sNB) * 100,          # -> %
      prevalence    = as.numeric(prevalence),
      outcome_label = factor(outcome_label_map_12[outcome], levels = panel_levels)
    ) %>%
    filter(!is.na(thresholds), !is.na(sNB))
}

dca_df_raw <- bind_rows(lapply(outcomes_12, read_dca_one))

# 5) Keep model + reference data, add labels
dca_models_raw <- dca_df_raw %>%
  filter(model_type %in% model_types_3) %>%
  mutate(
    model = model_type_labels[model_type],
    model = factor(model, levels = model_levels_3)
  )

dca_ref <- dca_df_raw %>%
  filter(model_type %in% c("treat_all", "treat_none")) %>%
  group_by(outcome_label, model_type) %>%
  arrange(thresholds) %>%
  ungroup()

# 6) Global threshold truncation based on model5
model5_type <- "model5_clinical_dynamic_exam"

thr_limits <- dca_models_raw %>%
  filter(model_type == model5_type) %>%
  group_by(outcome_label) %>%
  arrange(thresholds) %>%
  summarise(
    thr_cut = suppressWarnings(min(thresholds[sNB <= 0], na.rm = TRUE)),
    thr_max_data = suppressWarnings(max(thresholds, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(
    thr_cut = ifelse(is.finite(thr_cut), thr_cut, thr_max_data),
    thr_max = pmin(thr_cut + 2, thr_max_data),
    thr_max = pmax(thr_max, 5)
  )

dca_models_raw <- dca_models_raw %>%
  left_join(thr_limits, by = "outcome_label") %>%
  filter(thresholds <= thr_max) %>%
  select(-thr_cut, -thr_max_data, -thr_max)

dca_ref <- dca_ref %>%
  left_join(thr_limits, by = "outcome_label") %>%
  filter(thresholds <= thr_max) %>%
  select(-thr_cut, -thr_max_data, -thr_max)

# 7) Floor at 0 + absorb at 0 for model curves
enforce_floor_absorb0 <- function(df) {
  df <- df %>% arrange(thresholds)
  s <- pmax(df$sNB, 0)
  first_zero <- which(s <= 0)[1]
  if (!is.na(first_zero)) {
    s[first_zero:length(s)] <- 0
  }
  df$sNB <- s
  df
}

dca_models <- dca_models_raw %>%
  group_by(outcome_label, model_type) %>%
  group_modify(~ enforce_floor_absorb0(.x)) %>%
  ungroup() %>%
  mutate(
    model = factor(model_type_labels[model_type], levels = model_levels_3)
  )

# 8) Ribbon between Model 4 and Model 5
dca_ribbon <- dca_models %>%
  filter(model %in% c("Model 4: Clin+Exam (baseline)",
                      "Model 5: Clin+Exam (longitudinal)")) %>%
  select(outcome_label, thresholds, model, sNB) %>%
  pivot_wider(names_from = model, values_from = sNB) %>%
  drop_na(`Model 4: Clin+Exam (baseline)`, `Model 5: Clin+Exam (longitudinal)`) %>%
  arrange(outcome_label, thresholds) %>%
  mutate(
    ymin = pmin(`Model 4: Clin+Exam (baseline)`, `Model 5: Clin+Exam (longitudinal)`),
    ymax = pmax(`Model 4: Clin+Exam (baseline)`, `Model 5: Clin+Exam (longitudinal)`)
  )

# 9) Theme (same style; remove main title)
theme_dca <- theme_classic(base_size = 11) +
  theme(
    plot.title = element_blank(),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10, color = "#1F2328"),
    
    axis.title.x = element_text(size = 11, face = "bold", color = "#1F2328", margin = margin(t = 6)),
    axis.title.y = element_text(size = 11, face = "bold", color = "#1F2328", margin = margin(r = 6)),
    axis.text = element_text(size = 9, color = "#1F2328"),
    
    axis.ticks = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line  = element_line(linewidth = 0.45, color = "#2B2B2B"),
    
    panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
    panel.grid.major.x = element_line(color = "#F5F7FA", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 9.5, color = "#1F2328"),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    
    plot.margin = margin(10, 12, 8, 10)
  )

# 10) Plot
treat_all_df  <- dca_ref %>% filter(model_type == "treat_all")
treat_none_df <- dca_ref %>% filter(model_type == "treat_none")

p_dca_geo <- ggplot() +
  # Treat-none
  geom_line(
    data = treat_none_df,
    aes(x = thresholds, y = sNB, group = outcome_label),
    color = "#6B7280",
    linewidth = 0.45,
    linetype = "solid",
    alpha = 0.95
  ) +
  # Treat-all
  geom_line(
    data = treat_all_df,
    aes(x = thresholds, y = sNB, group = outcome_label),
    color = "#9CA3AF",
    linewidth = 0.45,
    linetype = "solid",
    alpha = 0.95
  ) +
  # Ribbon between Model 4 and Model 5
  geom_ribbon(
    data = dca_ribbon,
    aes(x = thresholds, ymin = ymin, ymax = ymax, group = outcome_label),
    fill = fill_color
  ) +
  # Model curves (ALL SOLID)
  geom_line(
    data = dca_models,
    aes(x = thresholds, y = sNB, color = model, group = interaction(outcome_label, model)),
    linewidth = 0.7,
    alpha = 0.95,
    linetype = "solid"
  ) +
  facet_wrap(~ outcome_label, nrow = 3, ncol = 4, scales = "free_x") +
  scale_color_manual(values = model_cols[model_levels_3]) +
  coord_cartesian(ylim = c(-1, NA)) +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.04))) +
  labs(
    x = "Threshold probability (%)",
    y = "Standardized net benefit (%)"
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_dca

p_dca_geo

ggsave("Fig5_geo_dca.pdf", p_dca_geo, height = 7, width = 9)






###### ukb roc 哑铃图
dat_ukb <- read_csv("XGBoost/all_models_summary_ukb_validation_year_10_pvalue.csv")

### model2 vs model5
suppressPackageStartupMessages({
  library(tidyverse)
  library(scales)
})

# 1) Disease labels (12)
outcome_label_map_12 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease"
)
outcomes_12 <- names(outcome_label_map_12)

# 2) Model labels & colors (same palette as you used before)
model_labels <- c(
  model2_clinical              = "Model 2: Clinical",
  model5_clinical_dynamic_exam = "Model 5: Clin+Exam (longitudinal)"
)

model_cols <- c(
  "Model 2: Clinical"                 = "#abd9e9",
  "Model 5: Clin+Exam (longitudinal)" = "#fc9272"
)

# 3) Prepare AUROC data (UKB external validation)
auc_df <- dat_ukb %>%
  filter(
    outcome %in% outcomes_12,
    model_type %in% names(model_labels)
  ) %>%
  transmute(
    outcome,
    outcome_label = factor(
      outcome_label_map_12[outcome],
      levels = unname(outcome_label_map_12)
    ),
    model = model_labels[model_type],
    auc   = as.numeric(ext_roc_auc),
    p_val = as.numeric(ext_vs_model5_auroc_pvalue)
  )

# 4) One row per outcome: wide for dumbbell endpoints
plot_df <- auc_df %>%
  select(outcome, outcome_label, model, auc) %>%
  pivot_wider(names_from = model, values_from = auc) %>%
  left_join(
    auc_df %>%
      filter(model == "Model 2: Clinical") %>%
      select(outcome, p_val),
    by = "outcome"
  ) %>%
  mutate(
    sig = case_when(
      is.na(p_val)  ~ "",
      p_val < 0.001 ~ "***",
      p_val < 0.01  ~ "**",
      p_val < 0.05  ~ "*",
      TRUE          ~ "ns"
    )
  )

# 5) Y-axis range (tight)
y_min <- min(plot_df$`Model 2: Clinical`,
             plot_df$`Model 5: Clin+Exam (longitudinal)`,
             na.rm = TRUE)
y_max <- max(plot_df$`Model 2: Clinical`,
             plot_df$`Model 5: Clin+Exam (longitudinal)`,
             na.rm = TRUE)

pad <- max(0.01, (y_max - y_min) * 0.06)  # slightly tighter than before

# 6) Put sig above the higher point
sig_y <- pmax(plot_df$`Model 2: Clinical`,
              plot_df$`Model 5: Clin+Exam (longitudinal)`,
              na.rm = TRUE) + pad * 0.35

# 7) Dumbbell plot
p_ukb_auc <- ggplot(plot_df, aes(x = outcome_label)) +
  # connecting line
  geom_segment(
    aes(
      xend = outcome_label,
      y    = `Model 2: Clinical`,
      yend = `Model 5: Clin+Exam (longitudinal)`
    ),
    color = "grey70",
    linewidth = 0.9
  ) +
  # Model 2 point
  geom_point(
    aes(y = `Model 2: Clinical`, color = "Model 2: Clinical"),
    size = 2.8
  ) +
  # Model 5 point
  geom_point(
    aes(y = `Model 5: Clin+Exam (longitudinal)`,
        color = "Model 5: Clin+Exam (longitudinal)"),
    size = 2.8
  ) +
  # significance (above)
  geom_text(
    aes(y = sig_y, label = sig),
    size = 3.2
  ) +
  scale_color_manual(values = model_cols) +
  scale_y_continuous(
    limits = c(y_min - pad, y_max + pad),
    breaks = pretty_breaks(5),
    labels = label_number(accuracy = 0.01)
  ) +
  labs(
    x = NULL,
    y = "AUROC"
  ) +
  theme_classic(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 9),
    axis.title.y = element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 9),
    plot.margin = margin(8, 10, 8, 8)
  )

p_ukb_auc

ggsave("Fig5_ukb_model2_vs_model5_auroc.pdf", p_ukb_auc, width = 6, height = 6)


### model4 vs model5
suppressPackageStartupMessages({
  library(tidyverse)
  library(scales)
})

# 1) Disease labels (12)
outcome_label_map_12 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease"
)
outcomes_12 <- names(outcome_label_map_12)

# 2) Model labels & colors (Model4 vs Model5)
model_labels <- c(
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

model_cols <- c(
  "Model 4: Clin+Exam (baseline)"     = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)" = "#fc9272"
)

# 3) Prepare AUROC data (UKB external validation)
auc_df <- dat_ukb %>%
  filter(
    outcome %in% outcomes_12,
    model_type %in% names(model_labels)
  ) %>%
  transmute(
    outcome,
    outcome_label = factor(
      outcome_label_map_12[outcome],
      levels = unname(outcome_label_map_12)
    ),
    model = model_labels[model_type],
    auc   = as.numeric(ext_roc_auc),
    p_val = as.numeric(ext_vs_model5_auroc_pvalue)
  )

# 4) One row per outcome (wide)
plot_df <- auc_df %>%
  select(outcome, outcome_label, model, auc) %>%
  pivot_wider(names_from = model, values_from = auc) %>%
  left_join(
    auc_df %>%
      filter(model == "Model 4: Clin+Exam (baseline)") %>%
      select(outcome, p_val),
    by = "outcome"
  ) %>%
  mutate(
    sig = case_when(
      is.na(p_val)  ~ "",
      p_val < 0.001 ~ "***",
      p_val < 0.01  ~ "**",
      p_val < 0.05  ~ "*",
      TRUE          ~ "ns"
    )
  )

# 5) Y-axis limits (tight)
y_min <- min(plot_df$`Model 4: Clin+Exam (baseline)`,
             plot_df$`Model 5: Clin+Exam (longitudinal)`,
             na.rm = TRUE)
y_max <- max(plot_df$`Model 4: Clin+Exam (baseline)`,
             plot_df$`Model 5: Clin+Exam (longitudinal)`,
             na.rm = TRUE)

pad <- max(0.01, (y_max - y_min) * 0.06)

# 6) Sig label y position: above the higher point
sig_y <- pmax(plot_df$`Model 4: Clin+Exam (baseline)`,
              plot_df$`Model 5: Clin+Exam (longitudinal)`,
              na.rm = TRUE) + pad * 0.35

# 7) Dumbbell plot
p_ukb_auc_m4m5 <- ggplot(plot_df, aes(x = outcome_label)) +
  geom_segment(
    aes(
      xend = outcome_label,
      y    = `Model 4: Clin+Exam (baseline)`,
      yend = `Model 5: Clin+Exam (longitudinal)`
    ),
    color = "grey70",
    linewidth = 0.9
  ) +
  geom_point(
    aes(y = `Model 4: Clin+Exam (baseline)`, color = "Model 4: Clin+Exam (baseline)"),
    size = 2.8
  ) +
  geom_point(
    aes(y = `Model 5: Clin+Exam (longitudinal)`, color = "Model 5: Clin+Exam (longitudinal)"),
    size = 2.8
  ) +
  geom_text(
    aes(y = sig_y, label = sig),
    size = 3.2
  ) +
  scale_color_manual(values = model_cols) +
  scale_y_continuous(
    limits = c(y_min - pad, y_max + pad),
    breaks = pretty_breaks(5),
    labels = label_number(accuracy = 0.01)
  ) +
  labs(x = NULL, y = "AUROC") +
  theme_classic(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 9),
    axis.title.y = element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 9),
    plot.margin = margin(8, 10, 8, 8)
  )

p_ukb_auc_m4m5

ggsave("Fig5_ukb_model4_vs_model5_auroc.pdf", p_ukb_auc_m4m5, width = 6, height = 6)



###### geo, ukb DR, 10% FDR
dat_geo_dr <- read_csv("XGBoost/DR_curve/dr_curve_fpr_grid_heldout.csv")
dat_ukb_dr <- read_csv("XGBoost/DR_curve/dr_curve_fpr_grid_ukb.csv")

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
})

# 1) Labels
outcome_label_map_12 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease"
)
outcomes_12 <- names(outcome_label_map_12)


# 2) Model tags & axis titles
model_tags <- list(
  m2 = "model2_clinical",
  m4 = "model4_clinical_baseline_exam",
  m5 = "model5_clinical_dynamic_exam"
)

axis_titles <- list(
  model2 = "DR (clinical)",
  model4 = "DR (clin + baseline exam)",
  model5 = "DR (clin + longitudinal exam)"
)


# 3) Disease colors
disease_cols <- c(
  "Myocardial infarction" = "#1B9E77",
  "Atrial fibrillation"   = "#D95F02",
  "Cor pulmonale"         = "#7570B3",
  "Heart failure"         = "#E7298A",
  "All-cause stroke"      = "#66A61E",
  "Haemorrhagic stroke"   = "#E6AB02",
  "COPD"                  = "#A6761D",
  "Liver cirrhosis"       = "#1F78B4",
  "Liver failure"         = "#B15928",
  "Renal failure"         = "#6A3D9A",
  "Diabetes"              = "#00897B",
  "Thyroid disease"       = "#8C6D31"
)


# 4) Prepare wide DR (pick FPR_grid=0.10) and convert DR*100
build_dr_wide <- function(dat, base_model_tag, new_model_tag, fpr_pick = 0.10) {
  
  dat2 <- dat %>%
    filter(
      outcome %in% outcomes_12,
      # 若有浮点误差可换成 abs(FPR_grid - fpr_pick) < 1e-8
      FPR_grid == fpr_pick,
      model %in% c(base_model_tag, new_model_tag)
    ) %>%
    mutate(
      outcome_label = factor(outcome_label_map_12[outcome],
                             levels = unname(outcome_label_map_12)),
      DR_pct = as.numeric(DR) * 100
    ) %>%
    select(outcome, outcome_label, model, DR_pct)
  
  wide <- dat2 %>%
    pivot_wider(names_from = model, values_from = DR_pct)
  
  # ensure columns exist
  if (!base_model_tag %in% names(wide)) wide[[base_model_tag]] <- NA_real_
  if (!new_model_tag  %in% names(wide)) wide[[new_model_tag]]  <- NA_real_
  
  wide
}


# 5) Plot function
plot_dr_scatter <- function(wide_df, x_col, y_col, x_title, y_title, show_legend = FALSE) {
  
  # axis limits tight with small padding, keep 1:1 view
  allv <- c(wide_df[[x_col]], wide_df[[y_col]])
  allv <- allv[is.finite(allv)]
  if (length(allv) == 0) {
    lims <- c(0, 1)
  } else {
    rng <- range(allv, na.rm = TRUE)
    pad <- (rng[2] - rng[1]) * 0.06
    if (!is.finite(pad) || pad == 0) pad <- 0.5
    lims <- c(rng[1] - pad, rng[2] + pad)
  }
  
  ggplot(wide_df, aes(x = .data[[x_col]], y = .data[[y_col]], color = outcome_label)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                linewidth = 0.6, color = "grey40") +
    geom_point(size = 3.0, alpha = 0.95) +
    scale_color_manual(values = disease_cols, drop = FALSE) +
    coord_cartesian(xlim = lims, ylim = lims) +
    labs(x = x_title, y = y_title) +
    theme_classic(base_size = 11) +
    theme(
      axis.title.x = element_text(face = "bold", size = 11, margin = margin(t = 6)),
      axis.title.y = element_text(face = "bold", size = 11, margin = margin(r = 6)),
      axis.text    = element_text(size = 9, color = "#1F2328"),
      axis.line    = element_line(linewidth = 0.45, color = "#2B2B2B"),
      axis.ticks   = element_line(linewidth = 0.35, color = "#2B2B2B"),
      panel.grid.major = element_line(color = "#F0F3F6", linewidth = 0.30),
      panel.grid.minor = element_blank(),
      legend.position = if (show_legend) "bottom" else "none",
      legend.title = element_blank(),
      legend.text  = element_text(size = 9),
      legend.key.height = unit(10, "pt"),
      legend.key.width  = unit(14, "pt"),
      plot.margin = margin(6, 6, 6, 6)
    ) +
    guides(color = guide_legend(nrow = 2, byrow = TRUE))
}


# 6) Build 4 panels
# Geo
geo_m2m5 <- build_dr_wide(dat_geo_dr, model_tags$m2, model_tags$m5, fpr_pick = 0.10)
geo_m4m5 <- build_dr_wide(dat_geo_dr, model_tags$m4, model_tags$m5, fpr_pick = 0.10)

# UKB
ukb_m2m5 <- build_dr_wide(dat_ukb_dr, model_tags$m2, model_tags$m5, fpr_pick = 0.10)
ukb_m4m5 <- build_dr_wide(dat_ukb_dr, model_tags$m4, model_tags$m5, fpr_pick = 0.10)

# Panels (no titles)
p1 <- plot_dr_scatter(
  geo_m2m5,
  x_col = model_tags$m2, y_col = model_tags$m5,
  x_title = axis_titles$model2, y_title = axis_titles$model5,
  show_legend = FALSE
)

p2 <- plot_dr_scatter(
  geo_m4m5,
  x_col = model_tags$m4, y_col = model_tags$m5,
  x_title = axis_titles$model4, y_title = axis_titles$model5,
  show_legend = FALSE
)

p3 <- plot_dr_scatter(
  ukb_m2m5,
  x_col = model_tags$m2, y_col = model_tags$m5,
  x_title = axis_titles$model2, y_title = axis_titles$model5,
  show_legend = FALSE
)

p4 <- plot_dr_scatter(
  ukb_m4m5,
  x_col = model_tags$m4, y_col = model_tags$m5,
  x_title = axis_titles$model4, y_title = axis_titles$model5,
  show_legend = TRUE
)

# 1 row × 4 cols, legend bottom 3×4
p_dr_4panel <- (p1 | p2 | p3 | p4) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom", legend.box = "horizontal")

p_dr_4panel


ggsave("Fig5_dr.pdf", p_dr_4panel, width = 13, height = 4)







##################### ------ Figure6, shap analysis ------
###### 全局shap热图
# 0) Config
OUTCOME_ENG_MAPPING <- c(
  "心肌梗死" = "mi",
  "心房颤动和扑动" = "afib_flutter",
  "肺心病" = "cor_pulmonale",
  "心力衰竭" = "chf",
  "中风" = "stroke",
  "缺血性中风" = "ischemic_stroke",
  "出血性中风" = "hemorrhagic_stroke",
  "动脉疾病" = "arterial_disease",
  "慢性阻塞性肺疾病" = "copd",
  "肝纤维化和肝硬化" = "liver_fibrosis_cirrhosis",
  "肝衰竭" = "liver_failure",
  "肾衰竭" = "renal_failure",
  "糖尿病" = "diabetes",
  "甲状腺疾病" = "thyroid_disease",
  "帕金森症" = "parkinson",
  "全因痴呆症" = "dementia",
  "泛癌" = "cancer_all",
  "肝癌" = "liver_cancer",
  "肺癌" = "lung_cancer",
  "肾癌" = "kidney_cancer"
)

outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)

GLOBAL_SHAP_DIR <- file.path("XGBoost", "shap_outputs", "global")

TOP_K_FEATURES <- 50
TARGET_COLOR   <- "#1F4E79"

# 1) Feature name standardizer

.standardize_feature_name_one <- function(x) {
  if (length(x) != 1 || is.na(x)) return(NA_character_)
  
  s <- x
  s <- gsub("_+", "_", s)
  s <- gsub("^_|_$", "", s)
  
  # mapping
  map <- c(
    "bmi" = "BMI",
    "waistcircumference" = "Waist Circ.",
    "bpsystolic" = "SBP",
    "bpdiastolic" = "DBP",
    "heartrate" = "Heart Rate",
    "hemoglobin" = "Hemoglobin",
    "wbc" = "WBC",
    "platelet" = "Platelet",
    "fastingglucosemmol" = "Fasting Glucose",
    "alt" = "ALT",
    "ast" = "AST",
    "totalbilirubin" = "Total Bilirubin",
    "creatinine" = "Creatinine",
    "serumurea" = "Urea",
    "totalcholesterol" = "TC",
    "triglycerides" = "TG",
    "ldl" = "LDL",
    "hdl" = "HDL",
    "sbp" = "SBP",
    "dbp" = "DBP",
    "tc" = "TC",
    "tg" = "TG"
  )
  
  s <- str_replace(s, "^mean_", "Mean_")
  s <- str_replace(s, "^sd_", "SD_")
  s <- str_replace(s, "^slope_", "Slope_")
  s <- str_replace(s, "^annual_ratio_change_", "Rel. change/yr_")
  s <- str_replace(s, "^annual_log_change_", "Log change/yr_")
  s <- str_replace(s, "^prop_high_", "Prop High_")
  
  tokens <- unlist(str_split(s, "_"))
  tokens <- tokens[tokens != ""]
  if (length(tokens) == 0) return(x)
  
  tokens_std <- vapply(tokens, function(tk) {
    if (str_detect(tk, " ")) return(str_to_title(tk))
    tk_low <- tolower(tk)
    if (tk_low %in% names(map)) return(map[[tk_low]])
    str_to_title(tk_low)
  }, FUN.VALUE = character(1))
  
  out <- paste(tokens_std, collapse = " ")
  
  # final acronym cleanup
  out <- str_replace_all(out, "\\bSbp\\b", "SBP")
  out <- str_replace_all(out, "\\bDbp\\b", "DBP")
  out <- str_replace_all(out, "\\bWbc\\b", "WBC")
  out <- str_replace_all(out, "\\bAlt\\b", "ALT")
  out <- str_replace_all(out, "\\bAst\\b", "AST")
  out <- str_replace_all(out, "\\bTg\\b", "TG")
  out <- str_replace_all(out, "\\bTc\\b", "TC")
  out <- str_replace_all(out, "\\bLdl\\b", "LDL")
  out <- str_replace_all(out, "\\bHdl\\b", "HDL")
  
  out
}

standardize_feature_name <- function(x) {
  # vectorized wrapper
  vapply(x, .standardize_feature_name_one, FUN.VALUE = character(1))
}

# 2) Read global shap files
read_one_global <- function(outcome_eng) {
  f <- file.path(GLOBAL_SHAP_DIR, paste0(outcome_eng, "_global_shap.csv"))
  if (!file.exists(f)) {
    message("[WARN] Missing: ", f)
    return(NULL)
  }
  read_csv(f, show_col_types = FALSE) %>%
    select(feature, mean_abs_shap) %>%
    mutate(disease = outcome_eng)
}

dfs <- purrr::map(OUTCOME_ENG_MAPPING, read_one_global) %>% purrr::compact()
if (length(dfs) == 0) stop("No global SHAP csv files found.")
df_all <- bind_rows(dfs)

# 3) Build matrix
heat_data <- df_all %>%
  pivot_wider(names_from = disease, values_from = mean_abs_shap, values_fill = 0) %>%
  as.data.frame()

rownames(heat_data) <- heat_data$feature
heat_data$feature <- NULL

# top K by cross-disease mean
overall <- rowMeans(as.matrix(heat_data))
top_features <- names(sort(overall, decreasing = TRUE))[1:min(TOP_K_FEATURES, length(overall))]
heat_top <- heat_data[top_features, , drop = FALSE]

# global max scale to [0,1]
max_val <- max(as.matrix(heat_top), na.rm = TRUE)
scaled_heat <- if (max_val == 0) heat_top else heat_top / max_val

# sort columns
desired_disease_ids <- names(outcome_label_map)
present <- intersect(desired_disease_ids, colnames(scaled_heat))
scaled_heat <- scaled_heat[, present, drop = FALSE]

# 4) Robust label maps
# disease labels in EXACT plotted order
col_ids <- colnames(scaled_heat)
col_labels <- outcome_label_map[col_ids] %>% as.character()
col_labels[is.na(col_labels)] <- col_ids[is.na(col_labels)]

# feature labels
feature_ids <- rownames(scaled_heat)
base_labels <- standardize_feature_name(feature_ids)

feature_map <- tibble(
  feature_id  = feature_ids,
  base_label  = base_labels
) %>%
  group_by(base_label) %>%
  mutate(feature_label = ifelse(n() == 1, base_label, paste0(base_label, " (", feature_id, ")"))) %>%
  ungroup() %>%
  mutate(
    # final safety: if still duplicates somehow, make.unique() as last resort
    feature_label = make.unique(feature_label)
  ) %>%
  select(feature_id, feature_label)

# hard guarantees
stopifnot(!any(is.na(feature_map$feature_label)))
stopifnot(!any(duplicated(feature_map$feature_label)))

# 5) Long df for ggplot2
df_plot <- as.data.frame(scaled_heat) %>%
  tibble::rownames_to_column("feature_id") %>%
  pivot_longer(-feature_id, names_to = "disease_id", values_to = "value") %>%
  left_join(feature_map, by = "feature_id") %>%
  mutate(
    value = replace_na(value, 0),
    disease_label = outcome_label_map[disease_id] %>% as.character(),
    disease_label = ifelse(is.na(disease_label), disease_id, disease_label)
  )

if (any(is.na(df_plot$feature_label))) stop("feature_label still has NA — this should not happen.")

# lock axis order
df_plot$disease_label <- factor(df_plot$disease_label, levels = col_labels)
df_plot$feature_label <- factor(df_plot$feature_label, levels = rev(feature_map$feature_label))

# 6) Plot
p <- ggplot(df_plot, aes(x = disease_label, y = feature_label, fill = value)) +
  geom_tile(color = NA) +
  scale_x_discrete(
    position = "top",
    guide = guide_axis(angle = 90)
  ) +
  scale_fill_gradient(
    low = "white",
    high = TARGET_COLOR,
    limits = c(0, 1),
    breaks = c(0, 0.25, 0.5, 0.75, 1),
    name = "Scaled global absolute SHAP value"
  ) +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid = element_blank(),
    axis.ticks = element_blank(),
    
    # 横纵轴字体更大
    axis.text.y = element_text(size = 10),
    axis.text.x.top = element_text(size = 10),
    
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    
    plot.margin = margin(4, 6, 4, 6)
  ) +
  guides(
    fill = guide_colorbar(
      title.position = "top",
      title.hjust = 0.5,
      barwidth = unit(7, "cm"),
      barheight = unit(0.45, "cm")
    )
  )

print(p)

ggsave("Fig6_shap_global_top_50.pdf", p, width = 6, height = 11)





###### 单个疾病shap图
# Config
outcome_label_map <- c(
  mi = "Myocardial infarction",
  afib_flutter = "Atrial fibrillation",
  cor_pulmonale = "Cor pulmonale",
  chf = "Heart failure",
  stroke = "All-cause stroke",
  ischemic_stroke = "Ischaemic stroke",
  hemorrhagic_stroke = "Haemorrhagic stroke",
  arterial_disease = "Arterial disease",
  copd = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure = "Liver failure",
  renal_failure = "Renal failure",
  diabetes = "Diabetes",
  thyroid_disease = "Thyroid disease",
  parkinson = "Parkinson's disease",
  dementia = "All-cause dementia",
  cancer_all = "All-cause cancer",
  liver_cancer = "Liver cancer",
  lung_cancer = "Lung cancer",
  kidney_cancer = "Kidney cancer"
)

GLOBAL_SHAP_DIR   <- "XGBoost/shap_outputs/global"
CIRCULAR_SHAP_DIR <- "XGBoost/shap_outputs/circular"

TOP_K <- 30

COLOR_LOW  <- "#2B6CB0"
COLOR_HIGH <- "#B40426"

# Feature name standardizer
.standardize_feature_name_one <- function(x) {
  if (length(x) != 1 || is.na(x)) return(NA_character_)
  
  s <- x
  s <- gsub("_+", "_", s)
  s <- gsub("^_|_$", "", s)
  
  map <- c(
    "bmi" = "BMI",
    "waistcircumference" = "Waist Circ.",
    "bpsystolic" = "SBP",
    "bpdiastolic" = "DBP",
    "heartrate" = "Heart Rate",
    "hemoglobin" = "Hemoglobin",
    "wbc" = "WBC",
    "platelet" = "Platelet",
    "fastingglucosemmol" = "Fasting Glucose",
    "alt" = "ALT",
    "ast" = "AST",
    "totalbilirubin" = "Total Bilirubin",
    "creatinine" = "Creatinine",
    "serumurea" = "Urea",
    "totalcholesterol" = "TC",
    "triglycerides" = "TG",
    "ldl" = "LDL",
    "hdl" = "HDL",
    "sbp" = "SBP",
    "dbp" = "DBP",
    "tc" = "TC",
    "tg" = "TG"
  )
  
  s <- str_replace(s, "^mean_", "Mean_")
  s <- str_replace(s, "^sd_", "SD_")
  s <- str_replace(s, "^slope_", "Slope_")
  s <- str_replace(s, "^annual_ratio_change_", "Rel. change/yr_")
  s <- str_replace(s, "^annual_log_change_", "Log change/yr_")
  s <- str_replace(s, "^prop_high_", "Prop High_")
  
  tokens <- unlist(str_split(s, "_"))
  tokens <- tokens[tokens != ""]
  if (length(tokens) == 0) return(x)
  
  tokens_std <- vapply(tokens, function(tk) {
    if (str_detect(tk, " ")) return(str_to_title(tk))
    tk_low <- tolower(tk)
    if (tk_low %in% names(map)) return(map[[tk_low]])
    str_to_title(tk_low)
  }, FUN.VALUE = character(1))
  
  out <- paste(tokens_std, collapse = " ")
  
  out <- str_replace_all(out, "\\bSbp\\b", "SBP")
  out <- str_replace_all(out, "\\bDbp\\b", "DBP")
  out <- str_replace_all(out, "\\bWbc\\b", "WBC")
  out <- str_replace_all(out, "\\bAlt\\b", "ALT")
  out <- str_replace_all(out, "\\bAst\\b", "AST")
  out <- str_replace_all(out, "\\bTg\\b", "TG")
  out <- str_replace_all(out, "\\bTc\\b", "TC")
  out <- str_replace_all(out, "\\bLdl\\b", "LDL")
  out <- str_replace_all(out, "\\bHdl\\b", "HDL")
  
  out
}
standardize_feature_name <- function(x) vapply(x, .standardize_feature_name_one, FUN.VALUE = character(1))

## Plot one disease
plot_circular_shap_rows <- function(outcome_eng, top_k = TOP_K) {
  
  circ_file <- file.path(CIRCULAR_SHAP_DIR, paste0(outcome_eng, "_circular_shap.csv"))
  glob_file <- file.path(GLOBAL_SHAP_DIR,   paste0(outcome_eng, "_global_shap.csv"))
  
  df_circ   <- readr::read_csv(circ_file, show_col_types = FALSE)
  df_global <- readr::read_csv(glob_file, show_col_types = FALSE)
  
  feat_order <- df_global %>%
    dplyr::arrange(dplyr::desc(mean_abs_shap)) %>%
    dplyr::slice_head(n = top_k) %>%
    dplyr::pull(feature) %>%
    unique()
  
  df <- df_circ %>% dplyr::filter(feature %in% feat_order)
  if (nrow(df) == 0) stop("No rows after filtering top features.")
  
  # Color value
  if ("normalized_feature_value" %in% names(df)) {
    df <- df %>% dplyr::mutate(norm_raw = normalized_feature_value)
  } else if ("mean_feature_value_z" %in% names(df)) {
    df <- df %>% dplyr::mutate(norm_raw = mean_feature_value_z)
  } else if ("mean_feature_value" %in% names(df)) {
    df <- df %>%
      dplyr::group_by(feature) %>%
      dplyr::mutate(
        norm_raw = {
          v <- mean_feature_value
          sdv <- stats::sd(v, na.rm = TRUE)
          ifelse(is.na(sdv) || sdv == 0, 0, (v - mean(v, na.rm = TRUE)) / sdv)
        }
      ) %>%
      dplyr::ungroup()
  } else {
    stop("Need normalized_feature_value / mean_feature_value_z / mean_feature_value for coloring.")
  }
  
  smax <- max(abs(df$norm_raw), na.rm = TRUE)
  if (!is.finite(smax) || smax == 0) smax <- 1
  df <- df %>% dplyr::mutate(norm = scales::squish(norm_raw / smax, range = c(-1, 1)))
  
  # stronger contrast
  gamma <- 0.45
  df <- df %>% dplyr::mutate(norm_boost = sign(norm) * (abs(norm) ^ gamma))
  
  # Feature labels + order
  feat_map <- tibble::tibble(feature = feat_order) %>%
    dplyr::mutate(feature_label = standardize_feature_name(feature))
  
  df <- df %>%
    dplyr::left_join(feat_map, by = "feature") %>%
    dplyr::mutate(feature_label = factor(feature_label, levels = rev(feat_map$feature_label)))
  
  # X-axis: mild truncation (99.7% abs)
  x_abs_lim <- stats::quantile(abs(df$mean_shap), probs = 0.997, na.rm = TRUE)
  if (!is.finite(x_abs_lim) || x_abs_lim == 0) x_abs_lim <- max(abs(df$mean_shap), na.rm = TRUE)
  x_abs_lim <- max(x_abs_lim, 0.25)
  
  df <- df %>% dplyr::mutate(mean_shap_clip = pmax(pmin(mean_shap, x_abs_lim), -x_abs_lim))
  x_breaks <- pretty(c(-x_abs_lim, x_abs_lim), n = 4)
  
  disease_label <- outcome_label_map[[outcome_eng]]
  if (is.null(disease_label) || is.na(disease_label)) disease_label <- outcome_eng

  # Plot
  p <- ggplot2::ggplot(df, ggplot2::aes(x = mean_shap_clip, y = feature_label)) +
    
    # points first
    ggplot2::geom_point(
      ggplot2::aes(color = norm_boost),
      size = 2.2,
      alpha = 1.0
    ) +
    
    ggplot2::geom_vline(
      xintercept = 0,
      color = "grey",
      linewidth = 0.7,
      linetype = "dashed"
    ) +
    
    ggplot2::scale_color_gradient2(
      low = COLOR_LOW, mid = "white", high = COLOR_HIGH,
      midpoint = 0,
      limits = c(-1, 1),
      breaks = c(-1, -0.5, 0, 0.5, 1),
      oob = scales::squish,
      name = "Normalized feature value"
    ) +
    
    ggplot2::scale_x_continuous(
      limits = c(-x_abs_lim, x_abs_lim),
      breaks = x_breaks
    ) +
    
    ggplot2::labs(
      title = disease_label,
      x = "Mean SHAP value",
      y = NULL
    ) +
    
    ggplot2::theme_minimal(base_size = 13) +
    ggplot2::theme(
      # remove grids
      panel.grid.major = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      
      axis.line.x = ggplot2::element_line(color = "black", linewidth = 0.7),
      axis.line.y = ggplot2::element_line(color = "black", linewidth = 0.7),
      
      axis.text.y = ggplot2::element_text(size = 12),
      axis.text.x = ggplot2::element_text(size = 12),
      
      plot.title  = ggplot2::element_text(face = "bold", size = 16, hjust = 0.5),
      
      legend.position  = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.title = ggplot2::element_text(size = 11),
      legend.text  = ggplot2::element_text(size = 10),
      legend.key.height = grid::unit(0.35, "cm"),
      legend.key.width  = grid::unit(0.9, "cm")
    ) +
    ggplot2::guides(
      color = ggplot2::guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = grid::unit(6.0, "cm"),
        barheight = grid::unit(0.35, "cm")
      )
    )
  
  return(p)
}

# 糖尿病
p <- plot_circular_shap_rows("diabetes", top_k = 30)
print(p)
ggsave("Fig6_diabetes_shap.pdf", p, height = 8, width = 9)

# 心肌梗塞
p <- plot_circular_shap_rows("mi", top_k = 30)
print(p)
ggsave("Fig6_mi_shap.pdf", p, height = 7, width = 9)

# 肝硬化
p <- plot_circular_shap_rows("liver_fibrosis_cirrhosis", top_k = 30)
print(p)
ggsave("Fig6_liver_fibrosis_cirrhosis_shap.pdf", p, height = 6, width = 9)

## 以下为附图
# 房颤
p <- plot_circular_shap_rows("afib_flutter", top_k = 30)
print(p)
ggsave("Ext_afib_flutter_shap.pdf", p, height = 8, width = 9)

# 肺心病
p <- plot_circular_shap_rows("cor_pulmonale", top_k = 30)
print(p)
ggsave("Ext_cor_pulmonale_shap.pdf", p, height = 8, width = 9)

# 心力衰竭
p <- plot_circular_shap_rows("chf", top_k = 30)
print(p)
ggsave("Ext_chf_shap.pdf", p, height = 8, width = 9)

# 中风
p <- plot_circular_shap_rows("stroke", top_k = 30)
print(p)
ggsave("Ext_stroke_shap.pdf", p, height = 8, width = 9)

# 缺血性中风
p <- plot_circular_shap_rows("ischemic_stroke", top_k = 30)
print(p)
ggsave("Ext_ischemic_stroke_shap.pdf", p, height = 8, width = 9)

# 出血性中风
p <- plot_circular_shap_rows("hemorrhagic_stroke", top_k = 30)
print(p)
ggsave("Ext_hemorrhagic_stroke_shap.pdf", p, height = 8, width = 9)

# 动脉疾病
p <- plot_circular_shap_rows("arterial_disease", top_k = 30)
print(p)
ggsave("Ext_arterial_disease_shap.pdf", p, height = 8, width = 9)

# 慢性阻塞性肺疾病
p <- plot_circular_shap_rows("copd", top_k = 30)
print(p)
ggsave("Ext_copd_shap.pdf", p, height = 8, width = 9)

# 肝衰竭
p <- plot_circular_shap_rows("liver_failure", top_k = 30)
print(p)
ggsave("Ext_liver_failure_shap.pdf", p, height = 7, width = 9)

# 肾衰竭
p <- plot_circular_shap_rows("renal_failure", top_k = 30)
print(p)
ggsave("Ext_renal_failure_shap.pdf", p, height = 8, width = 9)

# 甲状腺疾病
p <- plot_circular_shap_rows("thyroid_disease", top_k = 30)
print(p)
ggsave("Ext_thyroid_disease_shap.pdf", p, height = 8, width = 9)

# 帕金森
p <- plot_circular_shap_rows("parkinson", top_k = 30)
print(p)
ggsave("Ext_parkinson_shap.pdf", p, height = 6, width = 9)

# 痴呆
p <- plot_circular_shap_rows("dementia", top_k = 30)
print(p)
ggsave("Ext_dementia_shap.pdf", p, height = 8, width = 9)

# 癌症
p <- plot_circular_shap_rows("cancer_all", top_k = 30)
print(p)
ggsave("Ext_cancer_all_shap.pdf", p, height = 6, width = 9)

# 肝癌
p <- plot_circular_shap_rows("liver_cancer", top_k = 30)
print(p)
ggsave("Ext_liver_cancer_shap.pdf", p, height = 6, width = 9)

# 肺癌
p <- plot_circular_shap_rows("lung_cancer", top_k = 30)
print(p)
ggsave("Ext_lung_cancer_shap.pdf", p, height = 7, width = 9)

# 肾癌
p <- plot_circular_shap_rows("kidney_cancer", top_k = 30)
print(p)
ggsave("Ext_kidney_cancer_shap.pdf", p, height = 8, width = 9)





###### 堆叠条形图
###### 1. 基础设置 ######
GLOBAL_SHAP_DIR <- file.path("XGBoost", "shap_outputs", "global")

outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)

outcomes <- names(outcome_label_map)

###### 2. 五类特征分类 ######
## 注意：这个顺序用于控制图例顺序，并保证 Mean level 在最上面
category_levels_5 <- c(
  "Mean level",
  "Threshold exceedance proportion",
  "Variability",
  "Slope (trend)",
  "Annualized rate of change"
)

categorize_feature <- function(x) {
  case_when(
    str_detect(x, "^mean_")      ~ "Mean level",
    str_detect(x, "^sd_")        ~ "Variability",
    str_detect(x, "^slope_")     ~ "Slope (trend)",
    str_detect(x, "^annual_")    ~ "Annualized rate of change",
    str_detect(x, "^prop_high_") ~ "Threshold exceedance proportion",
    TRUE ~ NA_character_
  )
}

###### 3. 五类映射到三个宏类别，用于排序和加边框 ######
macro_category_map <- c(
  "Threshold exceedance proportion" = "Sustained level",
  "Mean level"                      = "Sustained level",
  "Variability"                     = "Variability",
  "Slope (trend)"                   = "Directional change",
  "Annualized rate of change"       = "Directional change"
)

macro_levels_3 <- c(
  "Sustained level",
  "Variability",
  "Directional change"
)

###### 4. 读取每个 outcome 的 global SHAP 并汇总 ######
process_one <- function(code) {
  
  file_path <- file.path(GLOBAL_SHAP_DIR, paste0(code, "_global_shap.csv"))
  if (!file.exists(file_path)) return(NULL)
  
  df <- readr::read_csv(file_path, show_col_types = FALSE) %>%
    mutate(category_5 = categorize_feature(feature)) %>%
    filter(!is.na(category_5))
  
  df %>%
    group_by(category_5) %>%
    summarise(value = mean(mean_abs_shap), .groups = "drop") %>%
    mutate(outcome = outcome_label_map[[code]])
}

shap_df <- purrr::map_dfr(outcomes, process_one)

###### 5. 计算五类占比 ######
shap_df <- shap_df %>%
  mutate(
    category_5 = factor(category_5, levels = category_levels_5),
    outcome    = factor(outcome, levels = unname(outcome_label_map))
  ) %>%
  complete(outcome, category_5, fill = list(value = 0)) %>%
  group_by(outcome) %>%
  mutate(prop = value / sum(value)) %>%
  ungroup()

###### 6. 计算三个宏类别占比，仅用于排序 ######
sort_df <- shap_df %>%
  mutate(
    macro_category = recode(as.character(category_5), !!!macro_category_map),
    macro_category = factor(macro_category, levels = macro_levels_3)
  ) %>%
  group_by(outcome, macro_category) %>%
  summarise(macro_prop = sum(prop), .groups = "drop") %>%
  pivot_wider(
    names_from  = macro_category,
    values_from = macro_prop,
    values_fill = 0
  ) %>%
  arrange(
    desc(`Sustained level`),
    desc(Variability),
    desc(`Directional change`)
  )

outcome_order <- sort_df$outcome

###### 7. 重新设定 outcome 顺序 ######
shap_df_plot <- shap_df %>%
  mutate(
    outcome    = factor(outcome, levels = outcome_order),
    category_5 = factor(category_5, levels = category_levels_5),
    macro_category = recode(as.character(category_5), !!!macro_category_map),
    macro_category = factor(macro_category, levels = macro_levels_3)
  )

###### 8. 计算每个柱子三大类边框的位置 ######
## 当前 geom_col() 的实际堆叠顺序是 factor levels 的反向
stack_order_bottom_to_top <- rev(category_levels_5)

## 给 outcome 分配数值型 x 坐标，便于 geom_rect 画边框
outcome_pos <- tibble(
  outcome = factor(outcome_order, levels = outcome_order),
  x_id = seq_along(outcome_order)
)

## 先按“实际堆叠顺序”计算每一小块的 ymin / ymax
stack_df <- shap_df_plot %>%
  left_join(outcome_pos, by = "outcome") %>%
  mutate(
    category_stack = factor(category_5, levels = stack_order_bottom_to_top)
  ) %>%
  arrange(outcome, category_stack) %>%
  group_by(outcome) %>%
  mutate(
    ymin = lag(cumsum(prop), default = 0),
    ymax = cumsum(prop)
  ) %>%
  ungroup()

## 再汇总成三大类边框
macro_rect_df <- stack_df %>%
  group_by(outcome, x_id, macro_category) %>%
  summarise(
    ymin = min(ymin),
    ymax = max(ymax),
    .groups = "drop"
  ) %>%
  mutate(
    xmin = x_id - 0.375,
    xmax = x_id + 0.375
  )

###### 9. 作图 ######
p <- ggplot(
  shap_df_plot,
  aes(
    x    = outcome,
    y    = prop,
    fill = category_5
  )
) +
  geom_col(width = 0.75) +
  
  ## 给三大类加边框
  geom_rect(
    data = macro_rect_df,
    aes(
      xmin = xmin,
      xmax = xmax,
      ymin = ymin,
      ymax = ymax
    ),
    inherit.aes = FALSE,
    fill = NA,
    colour = "black",
    linewidth = 0.35
  ) +
  
  scale_fill_manual(
    values = c(
      "Mean level"                      = "#8FA6D6",
      "Threshold exceedance proportion" = "#D9D9D9",
      "Variability"                     = "#B9CAE6",
      "Slope (trend)"                   = "#C8B7D8",
      "Annualized rate of change"       = "#E3CDBD"
    ),
    breaks = category_levels_5,
    drop = FALSE
  ) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    x = NULL,
    y = "Proportion of category-level importance",
    fill = NULL
  ) +
  guides(
    fill = guide_legend(
      ncol = 1,
      byrow = TRUE
    )
  ) +
  theme_classic(base_size = 16) +
  theme(
    axis.text.x = element_text(
      angle = 90,
      vjust = 0.5,
      hjust = 1
    ),
    legend.position   = "right",
    legend.direction  = "vertical",
    legend.box        = "vertical",
    legend.text       = element_text(size = 14),
    legend.spacing.y  = unit(6, "pt"),
    legend.key.height = unit(10, "pt"),
    legend.key.width  = unit(12, "pt")
  )

p

ggsave("Fig6e_shap_contribution_reordered_fixed_boxed.pdf", p, width = 16, height = 6)









##################### ------ Extended Data Table 1，Descriptive statistics ------
### 连续型变量
dat_train_con_desc <- read_csv("Desc/main/train_desc_continuous_summary.csv")
dat_test_con_desc <- read_csv("Desc/main/test_desc_continuous_summary.csv")

dat_train_con_desc <- dat_train_con_desc[1:19,]
name_map <- c(
  BMI = "BMI (kg/m²)",
  waistcircumference = "Waist circumference (cm)",
  bpsystolic = "Systolic blood pressure (mmHg)",
  bpdiastolic = "Diastolic blood pressure (mmHg)",
  respiratoryrate = "Respiratory rate (breaths/min)",
  heartrate = "Heart rate (beats/min)",
  hemoglobin = "Hemoglobin (g/L)",
  Wbc = "White blood cell count (×10⁹/L)",
  platelet = "Platelet count (×10⁹/L)",
  fastingglucosemmol = "Fasting glucose (mmol/L)",
  ALT = "Alanine aminotransferase (U/L)",
  AST = "Aspartate aminotransferase (U/L)",
  totalbilirubin = "Total bilirubin (µmol/L)",
  creatinine = "Creatinine (µmol/L)",
  serumurea = "Urea (mmol/L)",
  totalcholesterol = "Total cholesterol (mmol/L)",
  triglycerides = "Triglycerides (mmol/L)",
  LDL = "LDL cholesterol (mmol/L)",
  HDL = "HDL cholesterol (mmol/L)"
)

dat_train_con_desc$variable_label <- name_map[dat_train_con_desc$variable]
dat_train_con_desc <- dat_train_con_desc %>%
  mutate(mean = case_when(
    variable=="bpsystolic" | variable=="bpdiastolic" | variable=="respiratoryrate" | variable=="heartrate" ~ round(mean,0),
    TRUE ~ round(mean, 2)
  )) %>%
  mutate(sd = case_when(
    variable=="bpsystolic" | variable=="bpdiastolic" | variable=="respiratoryrate" | variable=="heartrate" ~ round(sd,1),
    TRUE ~ round(sd, 2)
  ))

dat_train_con_desc <- dat_train_con_desc %>%
  mutate(desc=paste0(as.character(mean), " (±", as.character(sd), ")"))

write_csv(dat_train_con_desc, "train_desc.csv")


dat_test_con_desc$variable_label <- name_map[dat_test_con_desc$variable]
dat_test_con_desc <- dat_test_con_desc %>%
  mutate(mean = case_when(
    variable=="bpsystolic" | variable=="bpdiastolic" | variable=="respiratoryrate" | variable=="heartrate" ~ round(mean,0),
    TRUE ~ round(mean, 2)
  )) %>%
  mutate(sd = case_when(
    variable=="bpsystolic" | variable=="bpdiastolic" | variable=="respiratoryrate" | variable=="heartrate" ~ round(sd,1),
    TRUE ~ round(sd, 2)
  ))

dat_test_con_desc <- dat_test_con_desc %>%
  mutate(desc=paste0(as.character(mean), " (±", as.character(sd), ")"))

write_csv(dat_test_con_desc, "test_desc.csv")


### 疾病统计
dat_train_dis_desc <- read_csv("Desc/main/train_desc_disease4_summary.csv")
dat_test_dis_desc <- read_csv("Desc/main/test_desc_disease4_summary.csv")

dat_train_dis_desc$note <- NULL
dat_test_dis_desc$note <- NULL

## 20 种疾病
target_diseases <- c(
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
)

## 疾病名称规范映射
disease_label_map <- c(
  mi = "Myocardial infarction",
  afib_flutter = "Atrial fibrillation",
  cor_pulmonale = "Cor pulmonale",
  chf = "Heart failure",
  stroke = "All-cause stroke",
  ischemic_stroke = "Ischaemic stroke",
  hemorrhagic_stroke = "Haemorrhagic stroke",
  arterial_disease = "Arterial disease",
  copd = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure = "Liver failure",
  renal_failure = "Renal failure",
  diabetes = "Diabetes",
  thyroid_disease = "Thyroid disease",
  parkinson = "Parkinson's disease",
  dementia = "All-cause dementia",
  cancer_all = "All-cause cancer",
  liver_cancer = "Liver cancer",
  lung_cancer = "Lung cancer",
  kidney_cancer = "Kidney cancer"
)

## 描述性统计表
dat_dis_desc_final <- dat_train_dis_desc %>%
  # 只保留目标疾病
  filter(variable %in% target_diseases) %>%
  filter(str_detect(level, "^1:")) %>%
  # 规范疾病名称 + 加 incidence
  mutate(
    Disease = paste0(disease_label_map[variable], " incidence"),
    # 格式化 n (%)
    `Cases, n (%)` = paste0(
      comma(n),
      " (",
      formatC(pct, format = "f", digits = 2),
      "%)"
    )
  ) %>%
  select(Disease, `Cases, n (%)`) %>%
  arrange(match(names(disease_label_map), Disease)) %>% 
  distinct()

write_csv(dat_dis_desc_final, "train_dis_desc.csv")

# test
dat_dis_desc_final <- dat_test_dis_desc %>%
  # 只保留目标疾病
  filter(variable %in% target_diseases) %>%
  filter(str_detect(level, "^1:")) %>%
  # 规范疾病名称 + 加 incidence
  mutate(
    Disease = paste0(disease_label_map[variable], " incidence"),
    # 格式化 n (%)
    `Cases, n (%)` = paste0(
      comma(n),
      " (",
      formatC(pct, format = "f", digits = 2),
      "%)"
    )
  ) %>%
  select(Disease, `Cases, n (%)`) %>%
  arrange(match(names(disease_label_map), Disease)) %>% 
  distinct()

write_csv(dat_dis_desc_final, "test_dis_desc.csv")



### UKB
dat_ukb <- read_feather("Desc/ukb/ukb_final_desc_data.feather")
df <- dat_ukb

## 描述性统计
## 0) 基础信息：总样本量 & city/cnty 唯一数
n_total <- nrow(df)

city_cnty_summary <- tibble(
  metric = c("N_total", "N_unique_city_raw", "N_unique_cnty_raw"),
  value  = c(
    n_total,
    dplyr::n_distinct(df$city_raw, na.rm = TRUE),
    dplyr::n_distinct(df$cnty_raw, na.rm = TRUE)
  )
)


## 1) 工具函数
pct <- function(x, denom) {
  100 * x / denom
}

# 单个分类变量：n 与 %
summarise_categorical <- function(data, var, denom = nrow(data)) {
  var <- rlang::ensym(var)
  data %>%
    mutate(.v = as.character(!!var)) %>%
    mutate(.v = ifelse(is.na(.v) | .v == "", "Missing", .v)) %>%
    count(.v, name = "n") %>%
    mutate(
      pct = pct(n, denom),
      variable = rlang::as_string(var),
      level = .v
    ) %>%
    select(variable, level, n, pct) %>%
    arrange(variable, level)
}

# one-hot 合并：prefix_1/prefix_2/... -> 一个类别变量
# 规则：哪个列为1就取该类别；多列同时为1 -> "Multiple"; 全0/全NA -> "Missing/Other"
summarise_onehot_group <- function(data, prefix, levels, denom = nrow(data)) {
  cols <- paste0(prefix, "_", levels)
  missing_cols <- setdiff(cols, names(data))
  if (length(missing_cols) > 0) {
    return(tibble(
      variable = prefix,
      level = NA_character_,
      n = NA_integer_,
      pct = NA_real_,
      note = paste0("Missing columns: ", paste(missing_cols, collapse = ", "))
    ))
  }
  
  mat <- data %>%
    select(all_of(cols)) %>%
    mutate(across(everything(), ~ suppressWarnings(as.integer(.)))) %>%
    as.matrix()
  
  # 计算每行命中的类别数与类别index
  hit_n <- rowSums(mat == 1, na.rm = TRUE)
  hit_idx <- apply(mat == 1, 1, function(r) {
    w <- which(r)
    if (length(w) == 1) w else NA_integer_
  })
  
  cat_val <- ifelse(hit_n == 0, "Missing/Other",
                    ifelse(hit_n > 1, "Multiple",
                           paste0(levels[hit_idx])))
  
  tab <- tibble(.v = cat_val) %>%
    count(.v, name = "n") %>%
    mutate(
      pct = pct(n, denom),
      variable = prefix,
      level = .v,
      note = ""
    ) %>%
    select(variable, level, n, pct, note) %>%
    arrange(variable, level)
  
  tab
}

# 疾病四分类变量：0/1/2/3 各自 n 与 %
summarise_disease4 <- function(data, var, denom = nrow(data)) {
  if (!(var %in% names(data))) {
    return(tibble(
      variable = var,
      level = NA_character_,
      n = NA_integer_,
      pct = NA_real_,
      note = "Missing column"
    ))
  }
  
  data %>%
    transmute(.v = as.integer(.data[[var]])) %>%
    mutate(.v = ifelse(is.na(.v), 99L, .v)) %>%
    mutate(
      level = case_when(
        .v == 0 ~ "0: no event in 4y (or after 4y)",
        .v == 1 ~ "1: incident in 4y",
        .v == 2 ~ "2: prevalent at/before baseline",
        .v == 3 ~ "3: died in 4y without event",
        TRUE    ~ "Missing/Other"
      )
    ) %>%
    count(level, name = "n") %>%
    mutate(
      pct = pct(n, denom),
      variable = var,
      note = ""
    ) %>%
    select(variable, level, n, pct, note) %>%
    arrange(variable, level)
}

# 连续变量：均值与标准差
summarise_continuous <- function(data, vars) {
  vars <- vars[vars %in% names(data)]
  if (length(vars) == 0) {
    return(tibble(variable = character(), mean = numeric(), sd = numeric(), n_nonmissing = integer()))
  }
  
  tibble(variable = vars) %>%
    rowwise() %>%
    mutate(
      x = list(suppressWarnings(as.numeric(data[[variable]]))),
      n_nonmissing = sum(!is.na(x)),
      mean = mean(unlist(x), na.rm = TRUE),
      sd = sd(unlist(x), na.rm = TRUE)
    ) %>%
    ungroup() %>%
    select(variable, n_nonmissing, mean, sd)
}


## 2) 分类变量：sex + one-hot组（合并后统计）
cat_tables <- list()

# sex
cat_tables$sex <- summarise_categorical(df, sex, denom = n_total)

# one-hot 组
cat_tables$parent_hyper    <- summarise_onehot_group(df, "parent_hyper",    levels = c(1,2,3), denom = n_total)
cat_tables$parent_diabetes <- summarise_onehot_group(df, "parent_diabetes", levels = c(1,2,3), denom = n_total)
cat_tables$smoke           <- summarise_onehot_group(df, "smoke",           levels = c(1,2,3), denom = n_total)
cat_tables$drink           <- summarise_onehot_group(df, "drink",           levels = c(1,2,3,4), denom = n_total)

categorical_summary <- bind_rows(cat_tables, .id = "group") %>%
  select(-group)


## 3) 疾病四分类变量：每个值(0/1/2/3)个数比例
disease_vars <- c(
  "hypertension","hyperlipidemia","mi","cor_pulmonale","cardiomyopathy","heart_block","afib_flutter",
  "chf","arterial_disease","stroke","ischemic_stroke","hemorrhagic_stroke","copd",
  "liver_fibrosis_cirrhosis","liver_failure","renal_failure","diabetes","thyroid_disease",
  "parkinson","dementia","cancer_all","liver_cancer","lung_cancer","kidney_cancer"
)

disease_summary <- map_dfr(disease_vars, ~ summarise_disease4(df, .x, denom = n_total))


## 4) 连续变量：均值、标准差
continuous_vars <- c(
  "BMI","waistcircumference","bpsystolic","bpdiastolic","respiratoryrate","heartrate",
  "hemoglobin","Wbc","platelet","fastingglucosemmol","ALT","AST","totalbilirubin",
  "creatinine","serumurea","totalcholesterol","triglycerides","LDL","HDL",
  "mean_BMI","sd_BMI","slope_BMI","annual_ratio_change_BMI","baseline_BMI",
  "mean_waistcircumference","sd_waistcircumference","slope_waistcircumference","annual_ratio_change_waistcircumference","baseline_waistcircumference",
  "mean_bpsystolic","sd_bpsystolic","slope_bpsystolic","annual_ratio_change_bpsystolic","baseline_bpsystolic",
  "mean_bpdiastolic","sd_bpdiastolic","slope_bpdiastolic","annual_ratio_change_bpdiastolic","baseline_bpdiastolic",
  "mean_respiratoryrate","sd_respiratoryrate","slope_respiratoryrate","annual_ratio_change_respiratoryrate","baseline_respiratoryrate",
  "mean_heartrate","sd_heartrate","slope_heartrate","annual_ratio_change_heartrate","baseline_heartrate",
  "mean_hemoglobin","sd_hemoglobin","slope_hemoglobin","annual_ratio_change_hemoglobin","baseline_hemoglobin",
  "mean_Wbc","sd_Wbc","slope_Wbc","annual_log_change_Wbc","baseline_Wbc",
  "mean_platelet","sd_platelet","slope_platelet","annual_log_change_platelet","baseline_platelet",
  "mean_fastingglucosemmol","sd_fastingglucosemmol","slope_fastingglucosemmol","annual_log_change_fastingglucosemmol","baseline_fastingglucosemmol",
  "mean_ALT","sd_ALT","slope_ALT","annual_log_change_ALT","baseline_ALT",
  "mean_AST","sd_AST","slope_AST","annual_log_change_AST","baseline_AST",
  "mean_totalbilirubin","sd_totalbilirubin","slope_totalbilirubin","annual_log_change_totalbilirubin","baseline_totalbilirubin",
  "mean_creatinine","sd_creatinine","slope_creatinine","annual_log_change_creatinine","baseline_creatinine",
  "mean_serumurea","sd_serumurea","slope_serumurea","annual_ratio_change_serumurea","baseline_serumurea",
  "mean_totalcholesterol","sd_totalcholesterol","slope_totalcholesterol","annual_ratio_change_totalcholesterol","baseline_totalcholesterol",
  "mean_triglycerides","sd_triglycerides","slope_triglycerides","annual_log_change_triglycerides","baseline_triglycerides",
  "mean_LDL","sd_LDL","slope_LDL","annual_ratio_change_LDL","baseline_LDL",
  "mean_HDL","sd_HDL","slope_HDL","annual_ratio_change_HDL","baseline_HDL",
  "prop_high_sbp","prop_high_dbp","prop_high_glucose","prop_high_TC","prop_high_TG","prop_high_LDL","prop_high_creatinine"
)

continuous_summary <- summarise_continuous(df, continuous_vars)


## 5) 导出
write_csv(categorical_summary, "ukb_desc_categorical_summary.csv")
write_csv(disease_summary, "ukb_desc_disease4_summary.csv")
write_csv(continuous_summary, "ukb_desc_continuous_summary.csv")

dat_train_con_desc <- continuous_summary[1:18,]
name_map <- c(
  BMI = "BMI (kg/m²)",
  waistcircumference = "Waist circumference (cm)",
  bpsystolic = "Systolic blood pressure (mmHg)",
  bpdiastolic = "Diastolic blood pressure (mmHg)",
  heartrate = "Heart rate (beats/min)",
  hemoglobin = "Hemoglobin (g/L)",
  Wbc = "White blood cell count (×10⁹/L)",
  platelet = "Platelet count (×10⁹/L)",
  fastingglucosemmol = "Fasting glucose (mmol/L)",
  ALT = "Alanine aminotransferase (U/L)",
  AST = "Aspartate aminotransferase (U/L)",
  totalbilirubin = "Total bilirubin (µmol/L)",
  creatinine = "Creatinine (µmol/L)",
  serumurea = "Urea (mmol/L)",
  totalcholesterol = "Total cholesterol (mmol/L)",
  triglycerides = "Triglycerides (mmol/L)",
  LDL = "LDL cholesterol (mmol/L)",
  HDL = "HDL cholesterol (mmol/L)"
)

dat_train_con_desc$variable_label <- name_map[dat_train_con_desc$variable]
dat_train_con_desc <- dat_train_con_desc %>%
  mutate(mean = case_when(
    variable=="bpsystolic" | variable=="bpdiastolic" | variable=="respiratoryrate" | variable=="heartrate" ~ round(mean,0),
    TRUE ~ round(mean, 2)
  )) %>%
  mutate(sd = case_when(
    variable=="bpsystolic" | variable=="bpdiastolic" | variable=="respiratoryrate" | variable=="heartrate" ~ round(sd,1),
    TRUE ~ round(sd, 2)
  ))

dat_train_con_desc <- dat_train_con_desc %>%
  mutate(desc=paste0(as.character(mean), " (±", as.character(sd), ")"))

write_csv(dat_train_con_desc, "ukb_desc.csv")


# 疾病统计
## 20 种疾病
target_diseases <- c(
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
)

## 疾病名称规范映射
disease_label_map <- c(
  mi = "Myocardial infarction",
  afib_flutter = "Atrial fibrillation",
  cor_pulmonale = "Cor pulmonale",
  chf = "Heart failure",
  stroke = "All-cause stroke",
  ischemic_stroke = "Ischaemic stroke",
  hemorrhagic_stroke = "Haemorrhagic stroke",
  arterial_disease = "Arterial disease",
  copd = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure = "Liver failure",
  renal_failure = "Renal failure",
  diabetes = "Diabetes",
  thyroid_disease = "Thyroid disease",
  parkinson = "Parkinson's disease",
  dementia = "All-cause dementia",
  cancer_all = "All-cause cancer",
  liver_cancer = "Liver cancer",
  lung_cancer = "Lung cancer",
  kidney_cancer = "Kidney cancer"
)

## 描述性统计表
dat_dis_desc_final <- disease_summary %>%
  # 只保留目标疾病
  filter(variable %in% target_diseases) %>%
  filter(str_detect(level, "^1:")) %>%
  # 规范疾病名称 + 加 incidence
  mutate(
    Disease = paste0(disease_label_map[variable], " incidence"),
    # 格式化 n (%)
    `Cases, n (%)` = paste0(
      comma(n),
      " (",
      formatC(pct, format = "f", digits = 2),
      "%)"
    )
  ) %>%
  select(Disease, `Cases, n (%)`) %>%
  arrange(match(names(disease_label_map), Disease)) %>% 
  distinct()

write_csv(dat_dis_desc_final, "ukb_dis_desc.csv")




### 连续变量发病组与健康组
df <- dat_ukb

## 1) 结局列表
outcome_vars <- c(
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
)


## 2) 连续变量列表（均值/SD）
continuous_vars <- c(
  "BMI","waistcircumference","bpsystolic","bpdiastolic","respiratoryrate","heartrate",
  "hemoglobin","Wbc","platelet","fastingglucosemmol","ALT","AST","totalbilirubin",
  "creatinine","serumurea","totalcholesterol","triglycerides","LDL","HDL",
  "mean_BMI","sd_BMI","slope_BMI","annual_ratio_change_BMI","baseline_BMI",
  "mean_waistcircumference","sd_waistcircumference","slope_waistcircumference","annual_ratio_change_waistcircumference","baseline_waistcircumference",
  "mean_bpsystolic","sd_bpsystolic","slope_bpsystolic","annual_ratio_change_bpsystolic","baseline_bpsystolic",
  "mean_bpdiastolic","sd_bpdiastolic","slope_bpdiastolic","annual_ratio_change_bpdiastolic","baseline_bpdiastolic",
  "mean_respiratoryrate","sd_respiratoryrate","slope_respiratoryrate","annual_ratio_change_respiratoryrate","baseline_respiratoryrate",
  "mean_heartrate","sd_heartrate","slope_heartrate","annual_ratio_change_heartrate","baseline_heartrate",
  "mean_hemoglobin","sd_hemoglobin","slope_hemoglobin","annual_ratio_change_hemoglobin","baseline_hemoglobin",
  "mean_Wbc","sd_Wbc","slope_Wbc","annual_log_change_Wbc","baseline_Wbc",
  "mean_platelet","sd_platelet","slope_platelet","annual_log_change_platelet","baseline_platelet",
  "mean_fastingglucosemmol","sd_fastingglucosemmol","slope_fastingglucosemmol","annual_log_change_fastingglucosemmol","baseline_fastingglucosemmol",
  "mean_ALT","sd_ALT","slope_ALT","annual_log_change_ALT","baseline_ALT",
  "mean_AST","sd_AST","slope_AST","annual_log_change_AST","baseline_AST",
  "mean_totalbilirubin","sd_totalbilirubin","slope_totalbilirubin","annual_log_change_totalbilirubin","baseline_totalbilirubin",
  "mean_creatinine","sd_creatinine","slope_creatinine","annual_log_change_creatinine","baseline_creatinine",
  "mean_serumurea","sd_serumurea","slope_serumurea","annual_ratio_change_serumurea","baseline_serumurea",
  "mean_totalcholesterol","sd_totalcholesterol","slope_totalcholesterol","annual_ratio_change_totalcholesterol","baseline_totalcholesterol",
  "mean_triglycerides","sd_triglycerides","slope_triglycerides","annual_log_change_triglycerides","baseline_triglycerides",
  "mean_LDL","sd_LDL","slope_LDL","annual_ratio_change_LDL","baseline_LDL",
  "mean_HDL","sd_HDL","slope_HDL","annual_ratio_change_HDL","baseline_HDL",
  "prop_high_sbp","prop_high_dbp","prop_high_glucose","prop_high_TC","prop_high_TG","prop_high_LDL","prop_high_creatinine"
)

continuous_vars <- intersect(continuous_vars, names(df))


## 3) 单结局：剔除2/3，只保留0/1，然后按组做均值/SD
summarise_cont_by_outcome01 <- function(data, outcome_var, cont_vars) {
  if (!(outcome_var %in% names(data))) {
    return(tibble(
      outcome = outcome_var, group = NA_character_, n_group = NA_integer_,
      variable = NA_character_, n = NA_integer_, mean = NA_real_, sd = NA_real_,
      note = "Outcome column missing"
    ))
  }
  
  d2 <- data %>%
    transmute(across(all_of(cont_vars), identity),
              .outcome = suppressWarnings(as.integer(.data[[outcome_var]]))) %>%
    # 只保留基线后发病(1)与基线后未发病(0)
    filter(.outcome %in% c(0L, 1L)) %>%
    mutate(group = ifelse(.outcome == 1L, "case(incident=1)", "noncase(0)"))
  
  # 每组人数
  n_group <- d2 %>% count(group, name = "n_group")
  
  # 连续变量统计
  res <- d2 %>%
    select(group, all_of(cont_vars)) %>%
    pivot_longer(cols = all_of(cont_vars), names_to = "variable", values_to = "value") %>%
    mutate(value = suppressWarnings(as.numeric(value))) %>%
    group_by(group, variable) %>%
    summarise(
      n = sum(!is.na(value)),
      mean = mean(value, na.rm = TRUE),
      sd = sd(value, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(outcome = outcome_var, note = "") %>%
    left_join(n_group, by = "group") %>%
    select(outcome, group, n_group, variable, n, mean, sd, note)
  
  res
}


## 4) 批量：20结局 × (0/1两组) × 连续变量
desc_by_outcome <- map_dfr(outcome_vars, ~ summarise_cont_by_outcome01(df, .x, continuous_vars))


## 5) 导出
write_csv(desc_by_outcome, "ukb_desc_continuous_by_outcome_incident1_vs_0.csv")










##################### ------ Extended Data Figure 2，Characteristic Distribution of Health Groups and Disease Groups ------
dat_train_full <- read_feather("Desc/XJ_desc/xj_train_desc_data.feather")
dat_test_full <- read_feather("Desc/XJ_desc/xj_test_desc_data.feather")

suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(grid)   # for unit()
})

df <- dat_train_full


## Outcomes used to define "disease-free controls"
## (controls are those with ALL of these == 0)
all_outcomes_20 <- c(
  "mi","afib_flutter","cor_pulmonale","chf","stroke","ischemic_stroke","hemorrhagic_stroke",
  "arterial_disease","copd","liver_fibrosis_cirrhosis","liver_failure","renal_failure","diabetes",
  "thyroid_disease","parkinson","dementia","cancer_all","liver_cancer","lung_cancer","kidney_cancer"
)


## Plot specifications (5 diseases × 4 features)
plot_specs <- list(
  mi = c("mean_bpsystolic","sd_bpsystolic","slope_bpsystolic","annual_ratio_change_bpsystolic"),
  hemorrhagic_stroke = c("mean_bpsystolic","sd_bpsystolic","slope_bpsystolic","annual_ratio_change_bpsystolic"),
  liver_fibrosis_cirrhosis = c("mean_AST","sd_AST","slope_AST","annual_log_change_AST"),
  renal_failure = c("mean_creatinine","sd_creatinine","slope_creatinine","annual_log_change_creatinine"),
  diabetes = c("mean_fastingglucosemmol","sd_fastingglucosemmol","slope_fastingglucosemmol","annual_log_change_fastingglucosemmol")
)


## Outcome labels
outcome_label_map <- c(
  mi = "Myocardial infarction",
  afib_flutter = "Atrial fibrillation",
  cor_pulmonale = "Cor pulmonale",
  chf = "Heart failure",
  stroke = "All-cause stroke",
  ischemic_stroke = "Ischaemic stroke",
  hemorrhagic_stroke = "Haemorrhagic stroke",
  arterial_disease = "Arterial disease",
  copd = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure = "Liver failure",
  renal_failure = "Renal failure",
  diabetes = "Diabetes",
  thyroid_disease = "Thyroid disease",
  parkinson = "Parkinson's disease",
  dementia = "All-cause dementia",
  cancer_all = "All-cause cancer",
  liver_cancer = "Liver cancer",
  lung_cancer = "Lung cancer",
  kidney_cancer = "Kidney cancer"
)


## Feature name standardizer
.standardize_feature_name_one <- function(x) {
  if (length(x) != 1 || is.na(x)) return(NA_character_)
  
  s <- x
  s <- gsub("_+", "_", s)
  s <- gsub("^_|_$", "", s)
  
  map <- c(
    "bmi" = "BMI",
    "waistcircumference" = "Waist Circ.",
    "bpsystolic" = "SBP",
    "bpdiastolic" = "DBP",
    "heartrate" = "Heart Rate",
    "hemoglobin" = "Hemoglobin",
    "wbc" = "WBC",
    "platelet" = "Platelet",
    "fastingglucosemmol" = "Fasting Glucose",
    "alt" = "ALT",
    "ast" = "AST",
    "totalbilirubin" = "Total Bilirubin",
    "creatinine" = "Creatinine",
    "serumurea" = "Urea",
    "totalcholesterol" = "TC",
    "triglycerides" = "TG",
    "ldl" = "LDL",
    "hdl" = "HDL",
    "sbp" = "SBP",
    "dbp" = "DBP",
    "tc" = "TC",
    "tg" = "TG"
  )
  
  s <- str_replace(s, "^mean_", "Mean_")
  s <- str_replace(s, "^sd_", "SD_")
  s <- str_replace(s, "^slope_", "Slope_")
  s <- str_replace(s, "^annual_ratio_change_", "Rel. change/yr_")
  s <- str_replace(s, "^annual_log_change_", "Log change/yr_")
  s <- str_replace(s, "^prop_high_", "Prop High_")
  
  tokens <- unlist(str_split(s, "_"))
  tokens <- tokens[tokens != ""]
  if (length(tokens) == 0) return(x)
  
  tokens_std <- vapply(tokens, function(tk) {
    if (str_detect(tk, " ")) return(str_to_title(tk))
    tk_low <- tolower(tk)
    if (tk_low %in% names(map)) return(map[[tk_low]])
    str_to_title(tk_low)
  }, FUN.VALUE = character(1))
  
  out <- paste(tokens_std, collapse = " ")
  
  out <- str_replace_all(out, "\\bSbp\\b", "SBP")
  out <- str_replace_all(out, "\\bDbp\\b", "DBP")
  out <- str_replace_all(out, "\\bWbc\\b", "WBC")
  out <- str_replace_all(out, "\\bAlt\\b", "ALT")
  out <- str_replace_all(out, "\\bAst\\b", "AST")
  out <- str_replace_all(out, "\\bTg\\b", "TG")
  out <- str_replace_all(out, "\\bTc\\b", "TC")
  out <- str_replace_all(out, "\\bLdl\\b", "LDL")
  out <- str_replace_all(out, "\\bHdl\\b", "HDL")
  
  out
}


## Group definition
for (v in all_outcomes_20) {
  if (!v %in% names(df)) stop("Missing outcome column in dat_train_full: ", v)
}

control_flag <- rep(TRUE, nrow(df))
for (v in all_outcomes_20) {
  control_flag <- control_flag & (suppressWarnings(as.integer(df[[v]])) == 0L)
}

## Colors
col_case <- "#B40426"
col_ctrl <- "#2B6CB0"


## Styling knobs
density_lw <- 0.85     
title_size <- 9       
legend_text_size <- 9
legend_title_size <- 10


## Single panel density plot
.make_density_plot <- function(case_df, ctrl_df, feature, outcome_key) {
  
  if (!feature %in% names(df)) {
    return(
      ggplot() +
        theme_void() +
        labs(title = .standardize_feature_name_one(feature)) +
        annotate("text", x = 0, y = 0, label = paste0("Missing: ", feature), hjust = 0)
    )
  }
  
  dd <- bind_rows(
    ctrl_df %>% transmute(group = "Control", value = suppressWarnings(as.numeric(.data[[feature]]))),
    case_df %>% transmute(group = "Case",    value = suppressWarnings(as.numeric(.data[[feature]])))
  ) %>%
    filter(!is.na(value)) %>%
    mutate(group = factor(group, levels = c("Control", "Case")))
  
  # counts shown in legend
  n_ctrl <- sum(dd$group == "Control")
  n_case <- sum(dd$group == "Case")
  
  legend_labels <- c(
    "Control" = paste0("Disease-free controls (n=", n_ctrl, ")"),
    "Case"    = paste0("Incident ", outcome_label_map[[outcome_key]], " (n=", n_case, ")")
  )
  
  ggplot(dd, aes(x = value, colour = group, fill = group)) +
    geom_density(alpha = 0.20, linewidth = density_lw, adjust = 1) +
    scale_colour_manual(
      values = c("Control" = col_ctrl, "Case" = col_case),
      breaks = c("Control", "Case"),
      labels = legend_labels
    ) +
    scale_fill_manual(
      values = c("Control" = col_ctrl, "Case" = col_case),
      breaks = c("Control", "Case"),
      labels = legend_labels
    ) +
    labs(
      title = .standardize_feature_name_one(feature),
      x = NULL,
      y = "Density",
      colour = "Group",
      fill = "Group"
    ) +
    theme_classic(base_size = 12) +
    theme(
      # (2) smaller + centered subplot titles
      plot.title = element_text(face = "bold", size = title_size, hjust = 0.5),
      axis.title.y = element_text(size = 10),
      axis.text = element_text(color = "black"),
      # (3) smaller legend
      legend.title = element_text(face = "bold", size = legend_title_size),
      legend.text = element_text(size = legend_text_size),
      legend.key.size = unit(0.35, "cm"),
      legend.spacing.y = unit(0.15, "cm"),
      legend.key = element_blank(),
      plot.margin = margin(6, 6, 6, 6)
    )
}


## One disease row: 1×4 + legend at bottom + disease title
.make_outcome_row <- function(outcome_key, features) {
  if (!outcome_key %in% names(df)) stop("Missing outcome column: ", outcome_key)
  
  case_df <- df %>%
    mutate(.y = suppressWarnings(as.integer(.data[[outcome_key]]))) %>%
    filter(.y == 1L)
  
  ctrl_df <- df %>%
    mutate(.ctrl = control_flag) %>%
    filter(.ctrl)
  
  p_list <- lapply(features, function(ft) .make_density_plot(case_df, ctrl_df, ft, outcome_key))
  
  row_plot <- wrap_plots(p_list, nrow = 1) +
    plot_layout(guides = "collect") &    # collect legend across 4 panels
    theme(legend.position = "bottom")    
  
  row_plot + plot_annotation(
    title = outcome_label_map[[outcome_key]]
  ) & theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0, margin = margin(b = 6))
  )
}


## Build 5×4 big figure
rows <- list(
  .make_outcome_row("mi", plot_specs$mi),
  .make_outcome_row("hemorrhagic_stroke", plot_specs$hemorrhagic_stroke),
  .make_outcome_row("liver_fibrosis_cirrhosis", plot_specs$liver_fibrosis_cirrhosis),
  .make_outcome_row("renal_failure", plot_specs$renal_failure),
  .make_outcome_row("diabetes", plot_specs$diabetes)
)

big_plot <- rows[[1]] / rows[[2]] / rows[[3]] / rows[[4]] / rows[[5]]

# Display
big_plot

ggsave("Ext_density_panel.pdf", big_plot, width = 12, height = 15)





##################### ------ Extended Data Figure 3，AUPRC ------
###### auprc绘图
dat_res <- read_csv("XGBoost/summary_metrics.csv")

# 1) outcome 显示名
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)
panel_levels <- unname(outcome_label_map)

# 2) 模型顺序与显示名（model1 -> model5）
model_order <- c(
  "model1_base",
  "model2_clinical",
  "model3_dynamic",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_label_map <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

# 3) 低饱和度配色
model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3",
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)

# 4) p值 -> 显著性标签
p_to_sig <- function(p) {
  ifelse(
    is.na(p), "",
    ifelse(p < 0.001, "***",
           ifelse(p < 0.01, "**",
                  ifelse(p < 0.05, "*", "ns")))
  )
}

# 5) 整理数据 + 计算标注高度
plot_dat <- dat_res %>%
  filter(model_type %in% model_order) %>%
  mutate(
    outcome_label = recode(outcome, !!!outcome_label_map),
    outcome_label = factor(outcome_label, levels = panel_levels), 
    model_type    = factor(model_type, levels = model_order),
    model_label   = factor(recode(as.character(model_type), !!!model_label_map),
                           levels = model_label_map[model_order]),
    x_num         = as.integer(model_type),
    sig_label     = ifelse(as.character(model_type) == "model5_clinical_dynamic_exam",
                           "ref",
                           p_to_sig(cv_vs_model5_auroc_pvalue))
  ) %>%
  select(
    outcome_label, model_label, x_num,
    cv_pr_auc_mean, cv_pr_auc_ci_lower, cv_pr_auc_ci_upper,
    sig_label
  ) %>%
  group_by(outcome_label) %>%
  mutate(
    y_span = max(cv_pr_auc_ci_upper, na.rm = TRUE) - min(cv_pr_auc_ci_lower, na.rm = TRUE),
    y_off  = ifelse(is.finite(y_span) & y_span > 0, 0.07 * y_span, 0.006),
    y_sig  = cv_pr_auc_ci_upper + y_off
  ) %>%
  ungroup()

# 6) 作图
p <- ggplot(plot_dat, aes(x = x_num, y = cv_pr_auc_mean)) +
  geom_line(aes(group = 1), linewidth = 0.55, color = "#C7CDD1") +
  geom_linerange(
    aes(ymin = cv_pr_auc_ci_lower, ymax = cv_pr_auc_ci_upper),
    linewidth = 0.55,
    color = "#3F4852"
  ) +
  geom_point(aes(color = model_label), size = 2.4) +
  geom_text(
    aes(y = y_sig, label = sig_label),
    size = 3.3,
    color = "#2F343A",
    vjust = 0
  ) +
  scale_color_manual(values = model_cols, name = NULL) +
  scale_x_continuous(
    breaks = 1:5,
    labels = rep("", 5),
    expand = expansion(mult = c(0.06, 0.06))
  ) +
  scale_y_continuous(
    breaks = pretty_breaks(n = 4),
    labels = label_number(accuracy = 0.001),
    expand = expansion(mult = c(0.06, 0.18))
  ) +
  labs(
    title = "Nested Cross-Validation AUPRC",
    x = NULL, y = NULL
  ) +
  ggh4x::facet_wrap2(
    ~ outcome_label,
    nrow = 4, ncol = 5,
    scales = "free_y",
    axes = "all"
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14, color = "#1F2328", hjust = 0.5),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10, color = "#1F2328"),
    
    axis.text.x = element_blank(),
    axis.ticks.x = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.text.y = element_text(size = 9, color = "#1F2328"),
    axis.ticks.y = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line = element_line(linewidth = 0.35, color = "#2B2B2B"),
    
    panel.grid.major.y = element_line(
      color = "#F4F6F8",
      linewidth = 0.3
    ),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.text = element_text(size = 9.5, color = "#1F2328"),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    
    plot.margin = margin(10, 12, 8, 10)
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(size = 3))) +
  coord_cartesian(clip = "off")

p

ggsave("Ext_AUPRC.pdf", p, width = 10, height = 7)







##################### ------ Extended Data Figure 4，performance of other ml models ------
###### AUROC
dat_CatBoost <- read_csv("CatBoost/summary_metrics.csv")
dat_LightGBM <- read_csv("LightGBM/summary_metrics.csv")
dat_Logistic <- read_csv("Logistic/summary_metrics.csv")

suppressPackageStartupMessages({
  library(tidyverse)
  library(scales)
  library(ggh4x)
  library(patchwork)
})


# 1) Outcome labels
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson’s disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)
outcomes_20 <- names(outcome_label_map)


panel_order <- unname(outcome_label_map) 


# 2) Algorithms + models
algo_levels <- c("CatBoost", "LightGBM", "Logistic")

model_type_labels <- c(
  model2_clinical               = "Model 2: Clinical",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)
model_levels <- unname(model_type_labels)

model_cols <- c(
  "Model 2: Clinical"                 = "#abd9e9",
  "Model 4: Clin+Exam (baseline)"     = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)" = "#fc9272"
)


# 3) Significance symbols
p_to_star <- function(p) {
  if (is.na(p)) return("")
  if (p < 1e-3) return("***")
  if (p < 1e-2) return("**")
  if (p < 5e-2) return("*")
  return("ns")  # set "" if you do not want ns
}


# 4) Combine algorithms
prep_one <- function(df, algo_label) {
  df %>%
    filter(outcome %in% outcomes_20,
           model_type %in% names(model_type_labels)) %>%
    transmute(
      outcome,
      algorithm = algo_label,
      model_type,
      model = factor(model_type_labels[model_type], levels = model_levels),
      AUROC = as.numeric(cv_roc_auc_mean),
      lo    = as.numeric(cv_roc_auc_ci_lower),
      hi    = as.numeric(cv_roc_auc_ci_upper),
      p_vs_m5 = as.numeric(cv_vs_model5_auroc_pvalue)
    )
}

plot_df <- bind_rows(
  prep_one(dat_CatBoost, "CatBoost"),
  prep_one(dat_LightGBM, "LightGBM"),
  prep_one(dat_Logistic, "Logistic")
) %>%
  mutate(
    outcome_label = factor(outcome_label_map[outcome], levels = panel_order),
    algorithm = factor(algorithm, levels = algo_levels),
    sig = case_when(
      model == "Model 5: Clin+Exam (longitudinal)" ~ "ref",
      TRUE ~ map_chr(p_vs_m5, p_to_star)
    )
  ) %>%
  filter(is.finite(AUROC), is.finite(lo), is.finite(hi))


# 5) Per-panel y limits + label height + floating-bar geometry
panel_limits <- plot_df %>%
  group_by(outcome_label) %>%
  summarise(
    ymin_raw = min(lo, na.rm = TRUE),
    ymax_raw = max(hi, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    rng  = pmax(0.02, ymax_raw - ymin_raw),
    ymin = ymin_raw - 0.20 * rng,
    ymax = ymax_raw + 0.35 * rng,
    y_text = ymax_raw + 0.08 * rng
  )

plot_df <- plot_df %>%
  left_join(panel_limits, by = "outcome_label") %>%
  mutate(
    algo_id  = as.numeric(algorithm),
    model_id = as.numeric(model)
  )

# floating bar widths
group_width <- 0.72
n_models <- length(model_levels)
step <- group_width / n_models
model_offset <- (-group_width/2) + (step/2) + (plot_df$model_id - 1) * step

plot_df <- plot_df %>%
  mutate(
    x_center = algo_id + model_offset,
    xmin = x_center - step * 0.45,
    xmax = x_center + step * 0.45
  )

# y scales per outcome
nice_y_breaks <- function(lim) {
  rng <- lim[2] - lim[1]
  step <- if (rng <= 0.05) 0.01 else if (rng <= 0.10) 0.02 else if (rng <= 0.20) 0.05 else 0.10
  seq(floor(lim[1] / step) * step, ceiling(lim[2] / step) * step, by = step)
}

y_scales_map <- panel_limits %>%
  mutate(
    y_scale = map2(ymin, ymax, ~{
      lim <- c(.x, .y)
      scale_y_continuous(
        limits = lim,
        breaks = nice_y_breaks(lim),
        labels = label_number(accuracy = 0.01),
        expand = expansion(mult = c(0, 0))
      )
    })
  ) %>%
  select(outcome_label, y_scale)

get_y_scales_for <- function(outcome_labels) {
  y_scales_map$y_scale[match(outcome_labels, y_scales_map$outcome_label)]
}


# 6) Theme
theme_base <- function(show_x = FALSE, show_y = FALSE) {
  theme_classic(base_size = 11) +
    theme(
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 10),
      
      axis.title.y = if (show_y) element_text(face = "bold", size = 11, margin = margin(r = 6))
      else element_blank(),
      
      # X axis text horizontal
      axis.text.x  = if (show_x) element_text(angle = 0, vjust = 1, hjust = 0.5, size = 9, color = "#1F2328")
      else element_blank(),
      axis.ticks.x = if (show_x) element_line(linewidth = 0.35, color = "#2B2B2B")
      else element_blank(),
      
      axis.text.y  = element_text(size = 9, color = "#1F2328"),
      axis.ticks.y = element_line(linewidth = 0.35, color = "#2B2B2B"),
      
      axis.line  = element_line(linewidth = 0.4, color = "#2B2B2B"),
      
      panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.3),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text  = element_text(size = 10),
      legend.key.width = unit(18, "pt"),
      legend.spacing.x = unit(10, "pt"),
      
      panel.spacing = unit(6, "pt"),
      plot.margin = margin(6, 10, 6, 8)
    )
}


# 7) Function: build one ROW (5 outcomes)
make_panel_row <- function(outcome_labels_row, show_x = FALSE, show_y = FALSE) {
  df_row <- plot_df %>% filter(outcome_label %in% outcome_labels_row) %>%
    mutate(outcome_label = factor(outcome_label, levels = outcome_labels_row))
  
  ggplot(df_row) +
    geom_rect(
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = AUROC, fill = model),
      color = NA, alpha = 0.95
    ) +
    geom_errorbar(
      aes(x = x_center, ymin = lo, ymax = hi),
      width = 0.06, linewidth = 0.35, color = "#1F2328"
    ) +
    geom_text(
      aes(x = x_center, y = y_text, label = sig),
      vjust = 0, size = 3.0, fontface = "bold", color = "#1F2328"
    ) +
    scale_fill_manual(values = model_cols) +
    scale_x_continuous(
      breaks = 1:3,
      labels = algo_levels,
      expand = expansion(mult = c(0.02, 0.02))
    ) +
    labs(
      y = if (show_y) "AUROC" else NULL,
      x = NULL
    ) +
    theme_base(show_x = show_x, show_y = show_y) +
    ggh4x::facet_wrap2(~ outcome_label, nrow = 1, ncol = 5, scales = "free", axes = "all") +
    ggh4x::facetted_pos_scales(y = get_y_scales_for(outcome_labels_row)) +
    coord_cartesian(clip = "off")
}


# 8) Build 4 rows × 5 cols
row1 <- panel_order[1:5]
row2 <- panel_order[6:10]
row3 <- panel_order[11:15]
row4 <- panel_order[16:20]

p1 <- make_panel_row(row1, show_x = FALSE, show_y = TRUE)
p2 <- make_panel_row(row2, show_x = FALSE, show_y = TRUE)
p3 <- make_panel_row(row3, show_x = FALSE, show_y = TRUE)
p4 <- make_panel_row(row4, show_x = TRUE,  show_y = TRUE)   # ONLY bottom row shows x

# Combine; collect legend at bottom
p_final <- (p1 / p2 / p3 / p4) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

p_final

ggsave("Ext_models_AUROC.pdf", p_final, width = 11, height = 7)





###### 校准曲线
suppressPackageStartupMessages({
  library(tidyverse)
  library(scales)
  library(ggh4x)
  library(grid)
})


# 0) Paths
pred_dirs <- c(
  "CatBoost"  = file.path("CatBoost",  "outer_cv_predictions"),
  "LightGBM"  = file.path("LightGBM",  "outer_cv_predictions"),
  "Logistic"  = file.path("Logistic",  "outer_cv_predictions")
)


# 1) Outcomes
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)
outcomes_20  <- names(outcome_label_map)
panel_levels <- unname(outcome_label_map)


# 2) Only Model 5
model5_type  <- "model5_clinical_dynamic_exam"
model5_label <- "Model 5: Clin+Exam (longitudinal)"


# 3) Colors (by algorithm)
algo_labels <- c(
  "CatBoost" = "CatBoost",
  "LightGBM" = "LightGBM",
  "Logistic" = "Logistic"
)

algo_cols <- c(
  "CatBoost"  = "#a1d99b",
  "LightGBM"  = "#fec44f",
  "Logistic"  = "#9ecae1"
)


# 4) Read prediction file (Model 5 only) from each algorithm folder
read_pred_one <- function(outcome, algo_key) {
  pred_dir <- pred_dirs[[algo_key]]
  f <- file.path(pred_dir, sprintf("%s_%s_outer5fold_predictions.csv", outcome, model5_type))
  if (!file.exists(f)) {
    warning("Missing prediction file: ", f)
    return(NULL)
  }
  readr::read_csv(f, show_col_types = FALSE) %>%
    transmute(
      outcome = outcome,
      algorithm = algo_key,
      actual = as.numeric(actual),
      pred   = as.numeric(pred_raw)
    ) %>%
    filter(!is.na(actual), !is.na(pred)) %>%
    mutate(pred = pmin(pmax(pred, 0), 1))
}

pred_df <- bind_rows(lapply(outcomes_20, function(oc) {
  bind_rows(lapply(names(pred_dirs), function(algo) read_pred_one(oc, algo)))
}))


# 5) Quantile-binned calibration (20 bins)
get_calib_bins <- function(df_one, n_bins = 20) {
  if (nrow(df_one) < n_bins) return(NULL)
  df_one %>%
    mutate(bin = ntile(pred, n_bins)) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(pred) * 100,
      obs_rate  = mean(actual) * 100,
      .groups = "drop"
    ) %>%
    arrange(mean_pred)
}

n_bins <- 20

calib_df <- pred_df %>%
  group_by(outcome, algorithm) %>%
  group_modify(~{
    out <- get_calib_bins(.x, n_bins = n_bins)
    if (is.null(out)) return(tibble())
    out
  }) %>%
  ungroup() %>%
  mutate(
    outcome_label = factor(outcome_label_map[outcome], levels = panel_levels),
    algorithm = factor(algo_labels[algorithm], levels = unname(algo_labels[names(pred_dirs)]))
  )


# 6) Better per-panel axis maxima
q_focus <- 0.99
min_cap <- 0.25
max_cap <- 100

panel_limits <- calib_df %>%
  group_by(outcome_label) %>%
  summarise(
    x_q = suppressWarnings(quantile(mean_pred, probs = q_focus, na.rm = TRUE)),
    y_q = suppressWarnings(quantile(obs_rate,  probs = q_focus, na.rm = TRUE)),
    x_max = max(mean_pred, na.rm = TRUE),
    y_max = max(obs_rate,  na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    m_q = pmax(ifelse(is.finite(x_q), x_q, x_max),
               ifelse(is.finite(y_q), y_q, y_max),
               na.rm = TRUE),
    m_max = m_q + pmax(0.15 * m_q, 0.05),
    m_max = pmin(pmax(m_max, min_cap), max_cap)
  )

make_breaks <- function(maxv) {
  step <- if (maxv <= 0.5) 0.1 else if (maxv <= 1) 0.2 else if (maxv <= 2) 0.5 else if (maxv <= 10) 2 else if (maxv <= 25) 5 else 10
  seq(0, maxv, by = step)
}
make_labels <- function(maxv) {
  if (maxv <= 1) label_number(accuracy = 0.1) else label_number(accuracy = 1)
}

x_scales <- lapply(panel_levels, function(lbl) {
  maxv <- panel_limits$m_max[match(lbl, panel_limits$outcome_label)]
  scale_x_continuous(
    limits = c(0, maxv),
    breaks = make_breaks(maxv),
    labels = make_labels(maxv),
    expand = expansion(mult = c(0.02, 0.04))
  )
})
y_scales <- lapply(panel_levels, function(lbl) {
  maxv <- panel_limits$m_max[match(lbl, panel_limits$outcome_label)]
  scale_y_continuous(
    limits = c(0, maxv),
    breaks = make_breaks(maxv),
    labels = make_labels(maxv),
    expand = expansion(mult = c(0.02, 0.04))
  )
})


# 7) Theme
theme_calib <- theme_classic(base_size = 11) +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10, color = "#1F2328"),
    axis.title.x = element_text(size = 11, face = "bold", color = "#1F2328", margin = margin(t = 6)),
    axis.title.y = element_text(size = 11, face = "bold", color = "#1F2328", margin = margin(r = 6)),
    axis.text = element_text(size = 9, color = "#1F2328"),
    axis.ticks = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line  = element_line(linewidth = 0.35, color = "#2B2B2B"),
    panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
    panel.grid.major.x = element_line(color = "#F5F7FA", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 10, color = "#1F2328"),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    plot.margin = margin(10, 12, 8, 10)
  )


# 8) Plot
p_calib_algo_m5 <- ggplot(calib_df, aes(x = mean_pred, y = obs_rate, color = algorithm, group = algorithm)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.35, color = "#8B949E") +
  geom_line(linewidth = 0.90, alpha = 0.95) +
  scale_color_manual(values = algo_cols) +
  labs(
    x = "Predicted risk (%)",
    y = "Observed event rate (%)"
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_calib +
  ggh4x::facet_wrap2(~ outcome_label, nrow = 4, ncol = 5, scales = "free", axes = "all") +
  ggh4x::facetted_pos_scales(x = x_scales, y = y_scales)

p_calib_algo_m5

ggsave("Ext_models_calib_model5.pdf", p_calib_algo_m5, width = 11, height = 7)















##################### ------ Extended Data Figure 5，result of full population ------
###### auroc绘图
dat_res <- read_csv("XGBoost/sen_fullpopulation/summary_metrics.csv")
# 1) outcome 显示名
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)
panel_levels <- unname(outcome_label_map)

# 2) 模型顺序与显示名（model1 -> model5）
model_order <- c(
  "model1_base",
  "model2_clinical",
  "model3_dynamic",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_label_map <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

# 3) 低饱和度配色
model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3",
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)

# 4) p值 -> 显著性标签
p_to_sig <- function(p) {
  ifelse(
    is.na(p), "",
    ifelse(p < 0.001, "***",
           ifelse(p < 0.01, "**",
                  ifelse(p < 0.05, "*", "ns")))
  )
}

# 5) 整理数据 + 计算标注高度
plot_dat <- dat_res %>%
  filter(model_type %in% model_order) %>%
  mutate(
    outcome_label = recode(outcome, !!!outcome_label_map),
    outcome_label = factor(outcome_label, levels = panel_levels), 
    model_type    = factor(model_type, levels = model_order),
    model_label   = factor(recode(as.character(model_type), !!!model_label_map),
                           levels = model_label_map[model_order]),
    x_num         = as.integer(model_type),
    sig_label     = ifelse(as.character(model_type) == "model5_clinical_dynamic_exam",
                           "ref",
                           p_to_sig(cv_vs_model5_auroc_pvalue))
  ) %>%
  select(
    outcome_label, model_label, x_num,
    cv_roc_auc_mean, cv_roc_auc_ci_lower, cv_roc_auc_ci_upper,
    sig_label
  ) %>%
  group_by(outcome_label) %>%
  mutate(
    y_span = max(cv_roc_auc_ci_upper, na.rm = TRUE) - min(cv_roc_auc_ci_lower, na.rm = TRUE),
    y_off  = ifelse(is.finite(y_span) & y_span > 0, 0.07 * y_span, 0.006),
    y_sig  = cv_roc_auc_ci_upper + y_off
  ) %>%
  ungroup()

# 6) 作图
p <- ggplot(plot_dat, aes(x = x_num, y = cv_roc_auc_mean)) +
  geom_line(aes(group = 1), linewidth = 0.55, color = "#C7CDD1") +
  geom_linerange(
    aes(ymin = cv_roc_auc_ci_lower, ymax = cv_roc_auc_ci_upper),
    linewidth = 0.55,
    color = "#3F4852"
  ) +
  geom_point(aes(color = model_label), size = 2.4) +
  geom_text(
    aes(y = y_sig, label = sig_label),
    size = 3.3,
    color = "#2F343A",
    vjust = 0
  ) +
  scale_color_manual(values = model_cols, name = NULL) +
  scale_x_continuous(
    breaks = 1:5,
    labels = rep("", 5),
    expand = expansion(mult = c(0.06, 0.06))
  ) +
  scale_y_continuous(
    breaks = pretty_breaks(n = 4),
    labels = label_number(accuracy = 0.001),
    expand = expansion(mult = c(0.06, 0.18))
  ) +
  labs(
    title = "Nested Cross-Validation AUROC",
    x = NULL, y = NULL
  ) +
  ggh4x::facet_wrap2(
    ~ outcome_label,
    nrow = 4, ncol = 5,
    scales = "free_y",
    axes = "all"
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14, color = "#1F2328", hjust = 0.5),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10, color = "#1F2328"),
    
    axis.text.x = element_blank(),
    axis.ticks.x = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.text.y = element_text(size = 9, color = "#1F2328"),
    axis.ticks.y = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line = element_line(linewidth = 0.35, color = "#2B2B2B"),
    
    panel.grid.major.y = element_line(
      color = "#F4F6F8",
      linewidth = 0.3
    ),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.text = element_text(size = 9.5, color = "#1F2328"),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    
    plot.margin = margin(10, 12, 8, 10)
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(size = 3))) +
  coord_cartesian(clip = "off")

p

ggsave("Ext_fullpopulation_AUROC.pdf", p, width = 10, height = 7)


###### 校准曲线
# 0) Paths
pred_dir <- "bench_dynamic/xgb/sensitivity/prediction_results/251209_final/outer_cv_predictions" 

# 1) Outcomes 
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)
outcomes_20  <- names(outcome_label_map)
panel_levels <- unname(outcome_label_map)

# 2) Models (5) 
model_types_5 <- c(
  "model1_base",
  "model2_clinical",
  "model3_dynamic",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

# 3) Colors
model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3",
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)

# 4) Read prediction file
read_pred_one <- function(outcome, model_type) {
  f <- file.path(pred_dir, sprintf("%s_%s_outer_cv_predictions.csv", outcome, model_type))
  if (!file.exists(f)) {
    warning("Missing prediction file: ", f)
    return(NULL)
  }
  readr::read_csv(f, show_col_types = FALSE) %>%
    transmute(
      outcome = outcome,
      model_type = model_type,
      actual = as.numeric(actual),
      pred   = as.numeric(pred_raw)
    ) %>%
    filter(!is.na(actual), !is.na(pred)) %>%
    mutate(pred = pmin(pmax(pred, 0), 1))
}

pred_df <- bind_rows(lapply(outcomes_20, function(oc) {
  bind_rows(lapply(model_types_5, function(mt) read_pred_one(oc, mt)))
}))

# 5) Quantile-binned calibration lines
get_calib_bins <- function(df_one, n_bins = 20) {
  if (nrow(df_one) < n_bins) return(NULL)
  
  df_one %>%
    mutate(bin = ntile(pred, n_bins)) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(pred) * 100,
      obs_rate  = mean(actual) * 100,
      .groups = "drop"
    ) %>%
    arrange(mean_pred)
}

n_bins <- 20

calib_df <- pred_df %>%
  group_by(outcome, model_type) %>%
  group_modify(~{
    out <- get_calib_bins(.x, n_bins = n_bins)
    if (is.null(out)) return(tibble())
    out
  }) %>%
  ungroup() %>%
  mutate(
    outcome_label = factor(outcome_label_map[outcome], levels = panel_levels),
    model = factor(model_type_labels[model_type],
                   levels = unname(model_type_labels[model_types_5]))
  )

# 6) Better per-panel axis maxima
q_focus <- 0.99          
min_cap <- 0.25          
max_cap <- 100

panel_limits <- calib_df %>%
  group_by(outcome_label) %>%
  summarise(
    x_q = suppressWarnings(quantile(mean_pred, probs = q_focus, na.rm = TRUE)),
    y_q = suppressWarnings(quantile(obs_rate,  probs = q_focus, na.rm = TRUE)),
    x_max = max(mean_pred, na.rm = TRUE),
    y_max = max(obs_rate,  na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    m_q = pmax(ifelse(is.finite(x_q), x_q, x_max),
               ifelse(is.finite(y_q), y_q, y_max),
               na.rm = TRUE),
    
    m_max = m_q + pmax(0.15 * m_q, 0.05),
    
    m_max = pmin(pmax(m_max, min_cap), max_cap)
  )

# 7) Panel-specific breaks/labels
make_breaks <- function(maxv) {
  # Choose step based on maxv
  step <- if (maxv <= 0.5) 0.1 else if (maxv <= 1) 0.2 else if (maxv <= 2) 0.5 else if (maxv <= 10) 2 else if (maxv <= 25) 5 else 10
  seq(0, maxv, by = step)
}

make_labels <- function(maxv) {
  # show one decimal for very small ranges
  if (maxv <= 1) label_number(accuracy = 0.1) else label_number(accuracy = 1)
}

x_scales <- lapply(panel_levels, function(lbl) {
  maxv <- panel_limits$m_max[match(lbl, panel_limits$outcome_label)]
  scale_x_continuous(
    limits = c(0, maxv),
    breaks = make_breaks(maxv),
    labels = make_labels(maxv),
    expand = expansion(mult = c(0.02, 0.04))
  )
})

y_scales <- lapply(panel_levels, function(lbl) {
  maxv <- panel_limits$m_max[match(lbl, panel_limits$outcome_label)]
  scale_y_continuous(
    limits = c(0, maxv),
    breaks = make_breaks(maxv),
    labels = make_labels(maxv),
    expand = expansion(mult = c(0.02, 0.04))
  )
})

# 8) Theme
theme_calib <- theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14, color = "#1F2328", hjust = 0.5),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10, color = "#1F2328"),
    
    axis.title.x = element_text(
      size = 11,
      face = "bold",        
      color = "#1F2328",
      margin = margin(t = 6)
    ),
    axis.title.y = element_text(
      size = 11,
      face = "bold",       
      color = "#1F2328",
      margin = margin(r = 6)
    ),
    axis.text = element_text(size = 9, color = "#1F2328"),
    axis.ticks = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line  = element_line(linewidth = 0.35, color = "#2B2B2B"),
    
    panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
    panel.grid.major.x = element_line(color = "#F5F7FA", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 9.5, color = "#1F2328"),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    
    plot.margin = margin(10, 12, 8, 10)
  )


# 9) Plot
p_calib <- ggplot(calib_df, aes(x = mean_pred, y = obs_rate, color = model, group = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.35, color = "#8B949E") +
  geom_line(linewidth = 0.85, alpha = 0.95) +
  scale_color_manual(values = model_cols) +
  labs(
    title = "Calibration Curve",
    x = "Predicted risk (%)",
    y = "Observed event rate (%)"
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_calib +
  ggh4x::facet_wrap2(~ outcome_label, nrow = 4, ncol = 5, scales = "free", axes = "all") +
  ggh4x::facetted_pos_scales(x = x_scales, y = y_scales)

p_calib

ggsave("Ext_fullpopulation_calibration.pdf", p_calib, width = 10, height = 7)





##################### ------ Extended Data Figure 6，clinical benefit，NRI，LR ------
###### NRI
dat_NRI <- read_csv("XGBoost/summary_metrics.csv")
dat_NRI <- select(dat_NRI, 1,2,14:22)
dat_NRI <- subset(dat_NRI, model_type=="model5_clinical_dynamic_exam")

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(scales)
})

# 1) Outcomes (12) + titles (standard)
outcome_label_map_12 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease"
)

outcomes_12 <- names(outcome_label_map_12)


# 2) Prepare NRI long data (Overall/Event/Non-event) + convert to percent
nri_long <- dat_NRI %>%
  filter(outcome %in% outcomes_12) %>%
  transmute(
    outcome,
    outcome_label = factor(outcome_label_map_12[outcome], levels = unname(outcome_label_map_12)),
    
    overall   = as.numeric(cv_model5_vs_model4_nri) * 100,
    overall_l = as.numeric(cv_model5_vs_model4_nri_lower) * 100,
    overall_u = as.numeric(cv_model5_vs_model4_nri_upper) * 100,
    
    event     = as.numeric(cv_model5_vs_model4_nri_event) * 100,
    event_l   = as.numeric(cv_model5_vs_model4_nri_event_lower) * 100,
    event_u   = as.numeric(cv_model5_vs_model4_nri_event_upper) * 100,
    
    nonevent   = as.numeric(cv_model5_vs_model4_nri_nonevent) * 100,
    nonevent_l = as.numeric(cv_model5_vs_model4_nri_nonevent_lower) * 100,
    nonevent_u = as.numeric(cv_model5_vs_model4_nri_nonevent_upper) * 100
  ) %>%
  pivot_longer(
    cols = c(overall, event, nonevent),
    names_to = "component",
    values_to = "nri"
  ) %>%
  mutate(
    lower = case_when(
      component == "overall"  ~ overall_l,
      component == "event"    ~ event_l,
      component == "nonevent" ~ nonevent_l
    ),
    upper = case_when(
      component == "overall"  ~ overall_u,
      component == "event"    ~ event_u,
      component == "nonevent" ~ nonevent_u
    ),
    component = recode(component,
                       overall  = "Overall",
                       event    = "Event",
                       nonevent = "Non-event"
    ),
    component = factor(component, levels = c("Overall", "Event", "Non-event"))
  ) %>%
  filter(is.finite(nri), is.finite(lower), is.finite(upper))


# 3) Colors
comp_cols <- c(
  "Overall"   = "#9e9ac8",
  "Event"     = "#fc9272",
  "Non-event" = "#6baed6"
)


# 4) Helper: per-outcome y limits
y_limits <- nri_long %>%
  group_by(outcome) %>%
  summarise(
    ymin = min(lower, nri, na.rm = TRUE),
    ymax = max(upper, nri, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    span = pmax(1, ymax - ymin),
    ymin2 = ymin - 0.10 * span,
    ymax2 = ymax + 0.12 * span,
    # ensure 0 line visible with some space
    ymin2 = pmin(ymin2, -0.5),
    ymax2 = pmax(ymax2,  0.5)
  )

get_ylim <- function(outcome_key) {
  i <- match(outcome_key, y_limits$outcome)
  c(y_limits$ymin2[i], y_limits$ymax2[i])
}

# nice y breaks
nice_breaks <- function(lims) {
  rng <- lims[2] - lims[1]
  step <- if (rng <= 5) 1 else if (rng <= 10) 2 else if (rng <= 20) 5 else 10
  seq(floor(lims[1] / step) * step, ceiling(lims[2] / step) * step, by = step)
}


# 5) Theme
theme_nri <- function() {
  theme_classic(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
      
      axis.title.x = element_blank(), # no x title
      axis.title.y = element_text(face = "bold", size = 11, margin = margin(r = 6)),
      
      axis.text.x  = element_blank(), # no x tick labels
      axis.ticks.x = element_blank(), # remove x ticks (but keep axis line)
      
      axis.text = element_text(size = 9, color = "#1F2328"),
      axis.ticks = element_line(linewidth = 0.35, color = "#2B2B2B"),
      axis.line  = element_line(linewidth = 0.45, color = "#2B2B2B"),
      
      panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
      panel.grid.major.x = element_line(color = "#F5F7FA", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 10),
      legend.key.width = unit(18, "pt"),
      legend.spacing.x = unit(10, "pt"),
      
      plot.margin = margin(6, 6, 6, 6)
    )
}


# 6) Build one panel plot
make_nri_panel <- function(outcome_key, show_y = FALSE) {
  df_one <- nri_long %>% filter(outcome == outcome_key)
  lims  <- get_ylim(outcome_key)
  
  ggplot(df_one, aes(x = component, y = nri, fill = component)) +
    geom_hline(yintercept = 0, linewidth = 0.35, color = "grey60") +
    geom_col(width = 0.72, alpha = 0.92) +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.10, linewidth = 0.45) +
    scale_fill_manual(values = comp_cols) +
    scale_y_continuous(
      limits = lims,
      breaks = nice_breaks(lims),
      labels = label_number(accuracy = 1),
      expand = expansion(mult = c(0.01, 0.03))
    ) +
    labs(
      title = outcome_label_map_12[[outcome_key]],
      y = if (show_y) "NRI (%)" else NULL
    ) +
    theme_nri() +
    theme(
      axis.title.y = if (show_y) element_text() else element_blank()
    )
}


# 7) 3x4 layout order & show y-title on left column only
outcomes_order <- outcomes_12

panels <- list()
for (i in seq_along(outcomes_order)) {
  oc <- outcomes_order[i]
  row <- ceiling(i / 4)
  col <- i - (row - 1) * 4
  show_y <- (col == 1)  # left column only (3 panels)
  panels[[i]] <- make_nri_panel(oc, show_y = show_y)
}

p_nri <- wrap_plots(panels, nrow = 3, ncol = 4) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

p_nri

ggsave("Ext_NRI.pdf", p_nri, width = 10, height = 7)








###### LR曲线
dat_LR <- read_csv("XGBoost/LR_Curve/lr_curve_fpr_grid.csv")

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(scales)
})


# 1) Outcomes (12) + titles (standard)
outcome_label_map_12 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease"
)

outcomes_12 <- names(outcome_label_map_12)


# 2) Models (3) + labels + colors
model_types_3 <- c(
  "model2_clinical",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

model_levels_3 <- unname(model_type_labels[model_types_3])

model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3",
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)


# 3) Prepare data: convert to % and standardize factors
lr_df <- dat_LR %>%
  filter(outcome %in% outcomes_12, model %in% model_types_3) %>%
  transmute(
    outcome,
    outcome_label = factor(outcome_label_map_12[outcome], levels = unname(outcome_label_map_12)),
    model_type = model,
    model = factor(model_type_labels[model_type], levels = model_levels_3),
    FDR = as.numeric(FPR_target) * 100,  # user wants label "FDR (%)"
    LR = as.numeric(LR_plus)
  ) %>%
  filter(!is.na(FDR), !is.na(LR)) %>%
  group_by(outcome, model) %>%
  arrange(FDR, .by_group = TRUE) %>%
  ungroup()


# 4) Helper: per-outcome y-axis max (dynamic, padded)
y_limits <- lr_df %>%
  group_by(outcome) %>%
  summarise(
    y_max_raw = max(LR, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    # Add padding; keep >= 5% headroom; handle near-zero
    y_max = pmax(5, y_max_raw * 1.10),
    y_max = ifelse(is.finite(y_max), y_max, 5)
  )

get_ymax <- function(outcome_key) {
  y_limits$y_max[match(outcome_key, y_limits$outcome)]
}

# Nice y breaks
nice_breaks <- function(maxv) {
  if (maxv <= 5)   return(seq(0, maxv, by = 1))
  if (maxv <= 10)  return(seq(0, maxv, by = 2))
  if (maxv <= 25)  return(seq(0, maxv, by = 5))
  if (maxv <= 50)  return(seq(0, maxv, by = 10))
  if (maxv <= 100) return(seq(0, maxv, by = 25))
  seq(0, maxv, by = 50)
}


# 5) Theme
theme_dr <- function() {
  theme_classic(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
      axis.title.x = element_text(face = "bold", size = 11, margin = margin(t = 6)),
      axis.title.y = element_text(face = "bold", size = 11, margin = margin(r = 6)),
      axis.text = element_text(size = 9, color = "#1F2328"),
      axis.ticks = element_line(linewidth = 0.35, color = "#2B2B2B"),
      axis.line  = element_line(linewidth = 0.45, color = "#2B2B2B"),
      panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
      panel.grid.major.x = element_line(color = "#F5F7FA", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 10),
      legend.key.width = unit(18, "pt"),
      legend.spacing.x = unit(10, "pt"),
      plot.margin = margin(6, 6, 6, 6)
    )
}


# 6) Build one panel plot
make_lr_panel <- function(outcome_key, show_x = FALSE, show_y = FALSE) {
  df_one <- dr_df %>% filter(outcome == outcome_key)
  ymax  <- get_ymax(outcome_key)
  
  ggplot(df_one, aes(x = FDR, y = LR, color = model, group = model)) +
    geom_line(linewidth = 0.95, linetype = "solid", alpha = 0.95) +
    geom_point(size = 1.3, alpha = 0.85) +
    scale_color_manual(values = model_cols[model_levels_3]) +
    scale_x_continuous(
      limits = c(0, 41),
      breaks = c(0, 10, 20, 30, 40),
      expand = expansion(mult = c(0.01, 0.02))
    ) +
    scale_y_continuous(
      limits = c(0, ymax),
      breaks = nice_breaks(ymax),
      labels = label_number(accuracy = 1),
      expand = expansion(mult = c(0.01, 0.03))
    ) +
    labs(
      title = outcome_label_map_12[[outcome_key]],
      x = if (show_x) "FDR (%)" else NULL,
      y = if (show_y) "LR" else NULL
    ) +
    theme_dr() +
    theme(
      # If not showing axis titles, also remove extra margin so panels align nicely
      axis.title.x = if (show_x) element_text() else element_blank(),
      axis.title.y = if (show_y) element_text() else element_blank()
    )
}


# 7) 3x4 layout order
outcomes_order <- outcomes_12 

panels <- list()
for (i in seq_along(outcomes_order)) {
  oc <- outcomes_order[i]
  row <- ceiling(i / 4)
  col <- i - (row - 1) * 4
  show_x <- (row == 3)          # bottom row
  show_y <- (col == 1)          # left column
  panels[[i]] <- make_lr_panel(oc, show_x = show_x, show_y = show_y)
}

# Combine and collect legend at bottom
p_lr <- wrap_plots(panels, nrow = 3, ncol = 4) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

p_lr

ggsave("Ext_LR.pdf", p_lr, width = 10, height = 7)





##################### ------ Extended Data Figure 7，subgroup analysis ------
###### AUROC
dat_subgroup_roc <- read_csv("XGBoost/subgroup_analysis/subgroup_performance.csv")
dat_subgroup_p <- read_csv("XGBoost/subgroup_analysis/subgroup_roc_pr_diff.csv")

suppressPackageStartupMessages({
  library(tidyverse)
  library(scales)
  library(patchwork)
})


# 1) Standard names
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)
outcomes_order <- names(outcome_label_map)

model_labels <- c(
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)
model_levels <- unname(model_labels)

model_cols <- c(
  "Model 4: Clin+Exam (baseline)"     = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)" = "#fc9272"
)

subgroup_label <- function(var, lvl) {
  case_when(
    var == "age_group" & lvl == "young_60" ~ "Age < 60",
    var == "age_group" & lvl == "old"      ~ "Age ≥ 60",
    var == "sex"       & lvl == "female"   ~ "Sex Female",
    var == "sex"       & lvl == "male"     ~ "Sex Male",
    var == "hypertension" & lvl == "0.0"   ~ "HTN No",
    var == "hypertension" & lvl == "1.0"   ~ "HTN Yes",
    TRUE ~ paste0(var, "\n", lvl)
  )
}

subgroup_levels_order <- c(
  "Age < 60", "Age ≥ 60",
  "Sex Female", "Sex Male",
  "HTN No", "HTN Yes"
)

# p -> stars
p_to_star <- function(p) {
  dplyr::case_when(
    is.na(p)      ~ "",
    p < 1e-3      ~ "***",
    p < 1e-2      ~ "**",
    p < 5e-2      ~ "*",
    TRUE          ~ "ns"
  )
}


# 2) Prepare data
roc_df <- dat_subgroup_roc %>%
  filter(model_type %in% names(model_labels)) %>%
  mutate(
    outcome_label = factor(outcome_label_map[outcome], levels = unname(outcome_label_map)),
    model = factor(model_labels[model_type], levels = model_levels),
    subgroup_lab = subgroup_label(subgroup_var, subgroup_level),
    subgroup_lab = factor(subgroup_lab, levels = subgroup_levels_order)
  ) %>%
  select(outcome, outcome_label, subgroup_var, subgroup_level, subgroup_lab,
         model, roc_auc, roc_auc_ci_lower, roc_auc_ci_upper)

p_df <- dat_subgroup_p %>%
  transmute(
    outcome,
    subgroup_var,
    subgroup_level,
    p_value = roc_p_value
  )

plot_df <- roc_df %>%
  left_join(p_df, by = c("outcome", "subgroup_var", "subgroup_level"))


# 3) Floating bar baseline per outcome
panel_limits <- plot_df %>%
  group_by(outcome) %>%
  summarise(
    y_min_ci = min(roc_auc_ci_lower, na.rm = TRUE),
    y_max_ci = max(roc_auc_ci_upper, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    rng_raw = y_max_ci - y_min_ci,
    rng = pmax(rng_raw, 0.012),
    
    # 更紧的 padding
    pad_low  = 0.04 * rng,
    pad_high = 0.06 * rng,
    
    y0    = pmax(0.50, y_min_ci - pad_low),
    y_max = pmin(1.00, y_max_ci + pad_high),
    
    y_txt = y_max_ci + 0.0025
  ) %>%
  mutate(
    y_max = pmin(1.00, pmax(y_max, y_txt + 0.003)),
    
    y_max = pmin(y_max, y0 + 0.18),
    
    y_max = pmin(1.00, pmax(y_max, y_txt + 0.003))
  )

plot_df <- plot_df %>%
  left_join(panel_limits, by = "outcome")


# 4) Build one panel
make_panel <- function(outcome_key, show_x = FALSE, show_y = FALSE) {
  
  df <- plot_df %>%
    filter(outcome == outcome_key) %>%
    mutate(
      x_id = as.numeric(subgroup_lab),
      m_id = as.numeric(model)
    )
  
  # bar geometry (2 bars per subgroup)
  group_width <- 0.62
  n_models <- length(model_levels)
  step <- group_width / n_models
  
  df <- df %>%
    mutate(
      x_center = x_id + (-group_width/2 + step/2) + (m_id - 1) * step,
      xmin = x_center - step * 0.45,
      xmax = x_center + step * 0.45
    )
  
  # label text
  df <- df %>%
    mutate(
      label = case_when(
        model == "Model 5: Clin+Exam (longitudinal)" ~ "ref",
        TRUE ~ p_to_star(p_value)
      )
    )
  
  ggplot(df) +
    # floating bars
    geom_rect(
      aes(xmin = xmin, xmax = xmax, ymin = y0, ymax = roc_auc, fill = model),
      alpha = 0.95,
      color = NA
    ) +
    # CI error bars
    geom_errorbar(
      aes(x = x_center, ymin = roc_auc_ci_lower, ymax = roc_auc_ci_upper),
      width = 0.05,
      linewidth = 0.35,
      color = "#1F2328"
    ) +
    # label above bars
    geom_text(
      aes(x = x_center, y = y_txt, label = label),
      size = 2.5,
      fontface = "bold",
      color = "#1F2328",
      vjust = 0.95
    ) +
    scale_fill_manual(values = model_cols) +
    scale_x_continuous(
      breaks = 1:length(subgroup_levels_order),
      labels = subgroup_levels_order,
      expand = expansion(mult = c(0.02, 0.02))
    ) +
    scale_y_continuous(
      limits = c(unique(df$y0), unique(df$y_max)),
      breaks = pretty_breaks(4),
      labels = label_number(accuracy = 0.01),
      expand = expansion(mult = c(0, 0))
    ) +
    labs(
      title = outcome_label_map[[outcome_key]],
      x = NULL,
      y = if (show_y) "AUROC" else NULL
    ) +
    theme_classic(base_size = 10) +
    theme(
      plot.title = element_text(face = "bold", size = 10, hjust = 0.5),
      
      axis.title.y = if (show_y) element_text(face = "bold", size = 11, margin = margin(r = 6)) else element_blank(),
      axis.text.y  = element_text(size = 8, color = "#1F2328"),
      axis.ticks.y = element_line(linewidth = 0.35, color = "#2B2B2B"),
      axis.line.y  = element_line(linewidth = 0.45, color = "#2B2B2B"),
      
      # x only on bottom row, vertical labels
      axis.text.x  = if (show_x) element_text(size = 8, angle = 90, vjust = 0.5, hjust = 1, color = "#1F2328") else element_blank(),
      axis.ticks.x = if (show_x) element_line(linewidth = 0.35, color = "#2B2B2B") else element_blank(),
      axis.line.x  = element_line(linewidth = 0.45, color = "#2B2B2B"),
      
      panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text  = element_text(size = 10),
      legend.key.width = unit(18, "pt"),
      legend.spacing.x = unit(10, "pt"),
      
      plot.margin = margin(6, 6, 6, 6)
    )
}


# 5) Assemble 4 rows × 5 cols
row1 <- outcomes_order[1:5]
row2 <- outcomes_order[6:10]
row3 <- outcomes_order[11:15]
row4 <- outcomes_order[16:20]

make_row <- function(row_outcomes, show_x = FALSE) {
  plots <- vector("list", length(row_outcomes))
  for (i in seq_along(row_outcomes)) {
    oc <- row_outcomes[i]
    show_y <- (i == 1)   # only leftmost in each row
    plots[[i]] <- make_panel(oc, show_x = show_x, show_y = show_y)
  }
  wrap_plots(plots, nrow = 1, ncol = 5)
}

p1 <- make_row(row1, show_x = FALSE)
p2 <- make_row(row2, show_x = FALSE)
p3 <- make_row(row3, show_x = FALSE)
p4 <- make_row(row4, show_x = TRUE)

p_final <- (p1 / p2 / p3 / p4) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

p_final

ggsave("Ext_subgroup_roc.pdf", p_final, width = 10, height = 7)



###### NRI
dat_subgroup_NRI <- read_csv("XGBoost/subgroup_analysis/subgroup_nri_idi.csv")
dat_subgroup_NRI <- subset(dat_subgroup_NRI, subgroup_var!="city_raw")

suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(scales)
})

# 1) Outcomes (12) + titles (standard)
outcome_label_map_12 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease"
)
outcomes_12 <- names(outcome_label_map_12)


# 2) Subgroup labels
subgroup_var_labels <- c(
  age_group    = "Age",
  sex          = "Sex",
  hypertension = "Hypertension"
)

subgroup_level_labels <- c(
  old      = ">=60",
  young_60 = "<60",
  male     = "Male",
  female   = "Female",
  `1.0`    = "Yes",
  `0.0`    = "No"
)

make_subgroup_label <- function(var, lvl) {
  var <- as.character(var)
  lvl <- as.character(lvl)
  
  dplyr::case_when(
    var == "age_group" & lvl == "young_60" ~ "Age < 60",
    var == "age_group" & lvl == "old"      ~ "Age >= 60",
    
    var == "sex" & lvl == "female"         ~ "Sex Female",
    var == "sex" & lvl == "male"           ~ "Sex Male",
    
    var == "hypertension" & lvl == "0.0"   ~ "HTN No",
    var == "hypertension" & lvl == "1.0"   ~ "HTN Yes",
    
    TRUE ~ paste0(var, "\n", lvl)
  )
}

subgroup_order <- c(
  "Age < 60","Age >= 60",
  "Sex Female","Sex Male",
  "HTN No","HTN Yes"
)


# 3) Prepare NRI long data
nri_long <- dat_subgroup_NRI %>%
  filter(outcome %in% outcomes_12) %>%
  transmute(
    outcome,
    outcome_label = factor(outcome_label_map_12[outcome], levels = unname(outcome_label_map_12)),
    subgroup_var,
    subgroup_level,
    subgroup = map2_chr(subgroup_var, subgroup_level, make_subgroup_label),
    # convert to %
    overall   = as.numeric(nri_new_vs_base) * 100,
    overall_l = as.numeric(nri_ci_lower) * 100,
    overall_u = as.numeric(nri_ci_upper) * 100,
    event     = as.numeric(nri_events) * 100,
    event_l   = as.numeric(nri_events_ci_lower) * 100,
    event_u   = as.numeric(nri_events_ci_upper) * 100,
    nonevent   = as.numeric(nri_nonevents) * 100,
    nonevent_l = as.numeric(nri_nonevents_ci_lower) * 100,
    nonevent_u = as.numeric(nri_nonevents_ci_upper) * 100
  ) %>%
  pivot_longer(
    cols = c(overall, event, nonevent),
    names_to = "component",
    values_to = "nri"
  ) %>%
  mutate(
    lower = case_when(
      component == "overall"  ~ overall_l,
      component == "event"    ~ event_l,
      component == "nonevent" ~ nonevent_l
    ),
    upper = case_when(
      component == "overall"  ~ overall_u,
      component == "event"    ~ event_u,
      component == "nonevent" ~ nonevent_u
    ),
    component = recode(component,
                       overall  = "Overall",
                       event    = "Event",
                       nonevent = "Non-event"),
    component = factor(component, levels = c("Overall","Event","Non-event")),
    subgroup  = factor(subgroup, levels = subgroup_order)
  ) %>%
  filter(is.finite(nri), is.finite(lower), is.finite(upper))


# 4) Colors
comp_cols <- c(
  "Overall"   = "#9e9ac8",
  "Event"     = "#fc9272",
  "Non-event" = "#6baed6"
)


# 5) Helper: per-outcome y limits
y_limits <- nri_long %>%
  group_by(outcome) %>%
  summarise(
    ymin = min(lower, nri, na.rm = TRUE),
    ymax = max(upper, nri, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    span = pmax(1, ymax - ymin),
    ymin2 = ymin - 0.08 * span,
    ymax2 = ymax + 0.10 * span,
    ymin2 = pmin(ymin2, -0.5),
    ymax2 = pmax(ymax2,  0.5)
  )

get_ylim <- function(outcome_key) {
  i <- match(outcome_key, y_limits$outcome)
  c(y_limits$ymin2[i], y_limits$ymax2[i])
}

nice_breaks <- function(lims) {
  rng <- lims[2] - lims[1]
  step <- if (rng <= 5) 1 else if (rng <= 10) 2 else if (rng <= 20) 5 else 10
  seq(floor(lims[1] / step) * step, ceiling(lims[2] / step) * step, by = step)
}


# 6) Theme
theme_nri <- function() {
  theme_classic(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
      
      axis.title.x = element_blank(),
      axis.title.y = element_text(face = "bold", size = 11, margin = margin(r = 6)),
      
      axis.text = element_text(size = 8, color = "#1F2328"),
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
      
      axis.ticks = element_line(linewidth = 0.35, color = "#2B2B2B"),
      axis.line  = element_line(linewidth = 0.45, color = "#2B2B2B"),
      
      panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 10),
      legend.key.width = unit(18, "pt"),
      legend.spacing.x = unit(10, "pt"),
      
      plot.margin = margin(6, 6, 6, 6)
    )
}


# 7) One disease panel
make_nri_panel <- function(outcome_key, show_y = FALSE, show_x = FALSE) {
  df_one <- nri_long %>% filter(outcome == outcome_key)
  lims  <- get_ylim(outcome_key)
  
  ggplot(df_one, aes(x = subgroup, y = nri, fill = component)) +
    geom_hline(yintercept = 0, linewidth = 0.35, color = "grey60") +
    geom_col(
      position = position_dodge(width = 0.70),
      width = 0.62,
      alpha = 0.92
    ) +
    geom_errorbar(
      aes(ymin = lower, ymax = upper),
      position = position_dodge(width = 0.70),
      width = 0.12,
      linewidth = 0.45
    ) +
    scale_fill_manual(values = comp_cols) +
    scale_y_continuous(
      limits = lims,
      breaks = nice_breaks(lims),
      labels = label_number(accuracy = 1),
      expand = expansion(mult = c(0.01, 0.03))
    ) +
    labs(
      title = outcome_label_map_12[[outcome_key]],
      y = if (show_y) "NRI (%)" else NULL
    ) +
    theme_nri() +
    theme(
      axis.title.y = if (show_y) element_text() else element_blank(),
      # only show x labels on bottom row
      axis.text.x  = if (show_x) element_text(angle = 90, vjust = 0.5, hjust = 1) else element_blank(),
      axis.ticks.x = if (show_x) element_line() else element_blank()
    )
}


# 8) 3x4 layout
panels <- list()
outcomes_order <- outcomes_12

for (i in seq_along(outcomes_order)) {
  oc <- outcomes_order[i]
  row <- ceiling(i / 4)
  col <- i - (row - 1) * 4
  show_y <- (col == 1)     # left column only
  show_x <- (row == 3)     # bottom row only
  panels[[i]] <- make_nri_panel(oc, show_y = show_y, show_x = show_x)
}

p_nri_subgroup <- wrap_plots(panels, nrow = 3, ncol = 4) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

p_nri_subgroup

ggsave("Ext_subgroup_NRI.pdf", p_nri_subgroup, width = 10, height = 7)










##################### ------ Extended Data Figure 8，sensitivity analysis, timewindows ------
dat_sen_pred_1yr <- read_csv("XGBoost/sen_timewindows/all_models_summary_sen_1yr.csv")
dat_sen_pred_2yr <- read_csv("XGBoost/sen_timewindows/all_models_summary_sen_2yr.csv")
dat_sen_pred_3yr <- read_csv("XGBoost/sen_timewindows/all_models_summary_sen_3yr.csv")
dat_sen_exclude_1mon <- read_csv("XGBoost/sen_timewindows/all_models_summary_sen_exclude_1mon.csv")
dat_sen_exclude_3mon <- read_csv("XGBoost/sen_timewindows/all_models_summary_sen_exclude_3mon.csv")
dat_sen_exclude_6mon <- read_csv("XGBoost/sen_timewindows/all_models_summary_sen_exclude_6mon.csv")
dat_sen_exclude_12mon <- read_csv("XGBoost/sen_timewindows/all_models_summary_sen_exclude_12mon.csv")

###### 预测时间窗口
suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(scales)
})

# 1) Outcome labels
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson’s disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)

outcomes_20 <- names(outcome_label_map)


# 2) Model labels & colors
model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

model_levels <- unname(model_type_labels)

model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3",
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)


# 3) Combine time-window data
make_one <- function(df, window_label) {
  df %>%
    transmute(
      outcome,
      model_type,
      window = window_label,
      AUROC = ext_roc_auc
    )
}

plot_df <- bind_rows(
  make_one(dat_sen_pred_1yr, "1-year"),
  make_one(dat_sen_pred_2yr, "2-year"),
  make_one(dat_sen_pred_3yr, "3-year")
) %>%
  filter(outcome %in% outcomes_20) %>%
  mutate(
    outcome_label = factor(outcome_label_map[outcome],
                           levels = unname(outcome_label_map)),
    model = factor(model_type_labels[model_type],
                   levels = model_levels),
    window = factor(window, levels = c("1-year", "2-year", "3-year"))
  )


# 4) Theme
theme_timewin <- function() {
  theme_classic(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
      
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 10),
      
      axis.text = element_text(size = 9, color = "#1F2328"),
      axis.ticks = element_line(linewidth = 0.35),
      axis.line  = element_line(linewidth = 0.4),
      
      panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.3),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 10),
      legend.key.width = unit(18, "pt"),
      legend.spacing.x = unit(10, "pt"),
      
      panel.spacing = unit(6, "pt"),
      plot.margin = margin(8, 10, 8, 8)
    )
}


# 5) Plot
p_timewin <- ggplot(
  plot_df,
  aes(x = window, y = AUROC, color = model, group = model)
) +
  geom_line(linewidth = 0.9, alpha = 0.95) +
  geom_point(size = 1.6, alpha = 0.95) +
  scale_color_manual(values = model_cols) +
  labs(
    y = "AUROC",
    x = NULL
  ) +
  facet_wrap(
    ~ outcome_label,
    nrow = 4,
    ncol = 5,
    scales = "free_y"
  ) +
  theme_timewin() +
  theme(
    axis.title.y = element_text(face = "bold"),
    axis.title.x = element_text(face = "bold")
  ) +
  guides(
    color = guide_legend(nrow = 1, byrow = TRUE)
  )

p_timewin

ggsave("Ext_sen_pred_timewindow.pdf", p_timewin, width = 10, height = 6)



###### 排除基线后一段时间的敏感性分析
suppressPackageStartupMessages({
  library(tidyverse)
  library(patchwork)
  library(scales)
})

# 1) Outcome labels
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson’s disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)

outcomes_20 <- names(outcome_label_map)


# 2) Model labels & colors
model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

model_levels <- unname(model_type_labels)

model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3",
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)


# 3) Combine exclusion-window data
make_one <- function(df, excl_label) {
  df %>%
    transmute(
      outcome,
      model_type,
      exclusion = excl_label,
      AUROC = ext_roc_auc
    )
}

plot_df_excl <- bind_rows(
  make_one(dat_sen_exclude_1mon,  "1-month"),
  make_one(dat_sen_exclude_3mon,  "3-month"),
  make_one(dat_sen_exclude_6mon,  "6-month"),
  make_one(dat_sen_exclude_12mon, "12-month")
) %>%
  filter(outcome %in% outcomes_20) %>%
  mutate(
    outcome_label = factor(outcome_label_map[outcome],
                           levels = unname(outcome_label_map)),
    model = factor(model_type_labels[model_type],
                   levels = model_levels),
    exclusion = factor(
      exclusion,
      levels = c("1-month", "3-month", "6-month", "12-month")
    )
  )


# 4) Theme
theme_timewin <- function() {
  theme_classic(base_size = 11) +
    theme(
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 10),
      
      axis.text.x = element_text(
        angle = 90,
        vjust = 0.5,
        hjust = 1,
        size = 9,
        color = "#1F2328"
      ),
      axis.text.y = element_text(size = 9, color = "#1F2328"),
      
      axis.ticks = element_line(linewidth = 0.35),
      axis.line  = element_line(linewidth = 0.4),
      
      panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.3),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 10),
      legend.key.width = unit(18, "pt"),
      legend.spacing.x = unit(10, "pt"),
      
      panel.spacing = unit(6, "pt"),
      plot.margin = margin(8, 10, 8, 8)
    )
}


# 5) Plot
p_exclusion <- ggplot(
  plot_df_excl,
  aes(x = exclusion, y = AUROC, color = model, group = model)
) +
  geom_line(linewidth = 0.9, alpha = 0.95) +
  geom_point(size = 1.6, alpha = 0.95) +
  scale_color_manual(values = model_cols) +
  labs(
    y = "AUROC",
    x = NULL
  ) +
  facet_wrap(
    ~ outcome_label,
    nrow = 4,
    ncol = 5,
    scales = "free_y"
  ) +
  theme_timewin() +
  guides(
    color = guide_legend(nrow = 1, byrow = TRUE)
  )

p_exclusion

ggsave("Ext_sen_exclude_timewindow.pdf", p_exclusion, width = 10, height = 7)







##################### ------ Supplementary Materials ------
###### Table S1 开发集、地理验证集、ukb基线疾病组与健康组特征的描述性统计
dat_dev <- read_csv("Desc/si/train_desc_continuous_by_outcome_incident1_vs_0.csv")
dat_test <- read_csv("Desc/si/test_desc_continuous_by_outcome_incident1_vs_0.csv")
dat_ukb <- read_csv("Desc/ukb/ukb_desc_continuous_by_outcome_incident1_vs_0.csv")

dat_dev$dataset <- "Development Set"
dat_test$dataset <- "Geographic External Validation Set"
dat_ukb$dataset <- "UKB External Validation Set"

dat_dataset_merge <- rbind(dat_dev, dat_test, dat_ukb)
dat_dataset_merge$note <- NULL
dat_dataset_merge <- select(dat_dataset_merge, 1,8,2,4:7)
names(dat_dataset_merge)[4] <- "var"

# 1) outcome (disease) mapping
outcome_label_map_20 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)

standardize_outcome_name <- function(x) {
  x <- as.character(x)
  out <- unname(outcome_label_map_20[x])
  ifelse(is.na(out), x, out)
}


# 2) var (feature) standardization
.standardize_feature_name_one <- function(x) {
  if (length(x) != 1 || is.na(x)) return(NA_character_)
  
  s <- as.character(x)
  
  # 0) normalize underscores & trim
  s <- gsub("_+", "_", s)
  s <- gsub("^_|_$", "", s)
  
  # 1) prefix -> pretty prefix (include baseline_)
  prefix_map <- c(
    "^baseline_"            = "Baseline_",
    "^mean_"                = "Mean_",
    "^sd_"                  = "SD_",
    "^slope_"               = "Slope_",
    "^annual_ratio_change_" = "Annual Ratio Change_",
    "^annual_log_change_"   = "Annual Log Change_",
    "^prop_high_"           = "Prop High_"
  )
  for (pat in names(prefix_map)) {
    s <- str_replace(s, pat, prefix_map[[pat]])
  }
  
  # 2) tokenization
  tokens <- unlist(str_split(s, "_"))
  tokens <- tokens[tokens != ""]
  if (length(tokens) == 0) return(x)
  
  # 3) biomarker & shorthand mapping (case-insensitive)
  map <- c(
    "bmi"                = "BMI",
    "waistcircumference" = "Waist Circ.",
    "bpsystolic"         = "SBP",
    "bpdiastolic"        = "DBP",
    "respiratoryrate"    = "Respiratory Rate",
    "heartrate"          = "Heart Rate",
    "hemoglobin"         = "Hemoglobin",
    "wbc"                = "WBC",
    "platelet"           = "Platelet",
    "fastingglucosemmol" = "Fasting Glucose",
    "alt"                = "ALT",
    "ast"                = "AST",
    "totalbilirubin"     = "Total Bilirubin",
    "creatinine"         = "Creatinine",
    "serumurea"          = "Urea",
    "totalcholesterol"   = "TC",
    "triglycerides"      = "TG",
    "ldl"                = "LDL",
    "hdl"                = "HDL",
    # for prop_high_*
    "sbp" = "SBP",
    "dbp" = "DBP",
    "tc"  = "TC",
    "tg"  = "TG"
  )
  
  # 4) standardize tokens
  tokens_std <- vapply(tokens, function(tk) {
    tk_chr <- as.character(tk)
    
    # keep multi-word phrases (from prefix_map) untouched
    if (str_detect(tk_chr, "\\s")) return(tk_chr)
    
    key <- tolower(tk_chr)
    if (key %in% names(map)) return(map[[key]])
    
    # default: Title Case
    str_to_title(key)
  }, FUN.VALUE = character(1))
  
  out <- paste(tokens_std, collapse = " ")
  
  # 5) final acronym cleanup
  out <- str_replace_all(out, "\\bSbp\\b", "SBP")
  out <- str_replace_all(out, "\\bDbp\\b", "DBP")
  out <- str_replace_all(out, "\\bWbc\\b", "WBC")
  out <- str_replace_all(out, "\\bAlt\\b", "ALT")
  out <- str_replace_all(out, "\\bAst\\b", "AST")
  out <- str_replace_all(out, "\\bTg\\b", "TG")
  out <- str_replace_all(out, "\\bTc\\b", "TC")
  out <- str_replace_all(out, "\\bLdl\\b", "LDL")
  out <- str_replace_all(out, "\\bHdl\\b", "HDL")
  
  out
}

standardize_feature_name <- function(x) {
  vapply(x, .standardize_feature_name_one, FUN.VALUE = character(1))
}


# 3) Apply to HR table
dat_dataset_merge <- dat_dataset_merge %>%
  mutate(
    outcome_std = standardize_outcome_name(outcome),
    var_std     = standardize_feature_name(var)
  )

dat_dataset_merge$outcome <- dat_dataset_merge$outcome_std
dat_dataset_merge$var <- dat_dataset_merge$var_std
dat_dataset_merge <- select(dat_dataset_merge, 1:7)

write_csv(dat_dataset_merge, "TableS1_case_control_desc.csv")




###### Table S2 开发集纵向特征Cox模型的HR结果
dat_table_hr <- read_csv("Epi/dynamic_feature_HR.csv")
dat_table_hr$note <- NULL
names(dat_table_hr)[c(1,2,9)] <- c("outcome", "var", "n_event")

# 1) outcome (disease) mapping
outcome_label_map_20 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)

standardize_outcome_name <- function(x) {
  x <- as.character(x)
  out <- unname(outcome_label_map_20[x])
  ifelse(is.na(out), x, out)
}


# 2) var (feature) standardization
.standardize_feature_name_one <- function(x) {
  if (length(x) != 1 || is.na(x)) return(NA_character_)
  
  s <- as.character(x)
  
  # 0) normalize underscores & trim
  s <- gsub("_+", "_", s)
  s <- gsub("^_|_$", "", s)
  
  # 1) prefix -> pretty prefix (include baseline_)
  prefix_map <- c(
    "^baseline_"            = "Baseline_",
    "^mean_"                = "Mean_",
    "^sd_"                  = "SD_",
    "^slope_"               = "Slope_",
    "^annual_ratio_change_" = "Annual Ratio Change_",
    "^annual_log_change_"   = "Annual Log Change_",
    "^prop_high_"           = "Prop High_"
  )
  for (pat in names(prefix_map)) {
    s <- str_replace(s, pat, prefix_map[[pat]])
  }
  
  # 2) tokenization
  tokens <- unlist(str_split(s, "_"))
  tokens <- tokens[tokens != ""]
  if (length(tokens) == 0) return(x)
  
  # 3) biomarker & shorthand mapping (case-insensitive)
  map <- c(
    "bmi"                = "BMI",
    "waistcircumference" = "Waist Circ.",
    "bpsystolic"         = "SBP",
    "bpdiastolic"        = "DBP",
    "respiratoryrate"    = "Respiratory Rate",
    "heartrate"          = "Heart Rate",
    "hemoglobin"         = "Hemoglobin",
    "wbc"                = "WBC",
    "platelet"           = "Platelet",
    "fastingglucosemmol" = "Fasting Glucose",
    "alt"                = "ALT",
    "ast"                = "AST",
    "totalbilirubin"     = "Total Bilirubin",
    "creatinine"         = "Creatinine",
    "serumurea"          = "Urea",
    "totalcholesterol"   = "TC",
    "triglycerides"      = "TG",
    "ldl"                = "LDL",
    "hdl"                = "HDL",
    # for prop_high_*
    "sbp" = "SBP",
    "dbp" = "DBP",
    "tc"  = "TC",
    "tg"  = "TG"
  )
  
  # 4) standardize tokens
  tokens_std <- vapply(tokens, function(tk) {
    tk_chr <- as.character(tk)
    
    # keep multi-word phrases (from prefix_map) untouched
    if (str_detect(tk_chr, "\\s")) return(tk_chr)
    
    key <- tolower(tk_chr)
    if (key %in% names(map)) return(map[[key]])
    
    # default: Title Case
    str_to_title(key)
  }, FUN.VALUE = character(1))
  
  out <- paste(tokens_std, collapse = " ")
  
  # 5) final acronym cleanup
  out <- str_replace_all(out, "\\bSbp\\b", "SBP")
  out <- str_replace_all(out, "\\bDbp\\b", "DBP")
  out <- str_replace_all(out, "\\bWbc\\b", "WBC")
  out <- str_replace_all(out, "\\bAlt\\b", "ALT")
  out <- str_replace_all(out, "\\bAst\\b", "AST")
  out <- str_replace_all(out, "\\bTg\\b", "TG")
  out <- str_replace_all(out, "\\bTc\\b", "TC")
  out <- str_replace_all(out, "\\bLdl\\b", "LDL")
  out <- str_replace_all(out, "\\bHdl\\b", "HDL")
  
  out
}

standardize_feature_name <- function(x) {
  vapply(x, .standardize_feature_name_one, FUN.VALUE = character(1))
}


# 3) Apply to HR table
dat_table_hr_std <- dat_table_hr %>%
  mutate(
    outcome_std = standardize_outcome_name(outcome),
    var_std     = standardize_feature_name(var)
  )

write_csv(dat_table_hr_std, "TableS2_HR_res_std.csv")





###### Table S3 XGBoost等4个模型的各项评价指标（嵌套交叉验证、地理外部验证）
dat_xgb <- read_csv("XGBoost/summary_metrics.csv")
dat_catboost <- read_csv("CatBoost/summary_metrics.csv")
dat_lgb <- read_csv("LightGBM/summary_metrics.csv")
dat_logistic <- read_csv("Logistic/summary_metrics.csv")

dat_xgb <- select(dat_xgb, -c(23:25,46:48))
dat_catboost <- select(dat_catboost, -c(23:25,46:48))
dat_lgb <- select(dat_lgb, -c(23:25,46:48))
dat_logistic <- select(dat_logistic, -c(23:25,46:48))

dat_xgb$model <- "XGBoost"
dat_catboost$model <- "CatBoost"
dat_lgb$model <- "LightGBM"
dat_logistic$model <- "Logistic"

dat_merge_res <- rbind(dat_xgb, dat_catboost, dat_lgb, dat_logistic)
dat_merge_res <- select(dat_merge_res, 1,43,2:42)


# 0) Standard name maps
# Disease standard names
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)

# Model standard names (5)
model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

# Prefixes you requested
cv_prefix   <- "nested_cross_validation_"
test_prefix <- "geographic_external_val_"


# 1) Rename columns: cv_ -> nested_cross_validation_ ; test_ -> geographic_external_val_
rename_metrics_cols <- function(df) {
  df %>%

    rename_with(~ str_replace_all(.x, "^cv_",   cv_prefix),   .cols = matches("^cv_")) %>%
    rename_with(~ str_replace_all(.x, "^test_", test_prefix), .cols = matches("^test_")) %>%
    

    rename_with(
      ~ str_replace_all(.x,
                        paste0("^", cv_prefix, "vs_model5_"),
                        paste0(cv_prefix, "vs_model5_")),
      .cols = matches(paste0("^", cv_prefix, "vs_model5_"))
    ) %>%
    

    rename_with(
      ~ str_replace_all(.x,
                        paste0("^", cv_prefix, "model5_vs_model4_"),
                        paste0(cv_prefix, "model5_vs_model4_")),
      .cols = matches(paste0("^", cv_prefix, "model5_vs_model4_"))
    ) %>%
    

    rename_with(
      ~ str_replace_all(.x,
                        paste0("^", test_prefix, "vs_model5_"),
                        paste0(test_prefix, "vs_model5_")),
      .cols = matches(paste0("^", test_prefix, "vs_model5_"))
    ) %>%
    

    rename_with(
      ~ str_replace_all(.x,
                        paste0("^", test_prefix, "model5_vs_model4_"),
                        paste0(test_prefix, "model5_vs_model4_")),
      .cols = matches(paste0("^", test_prefix, "model5_vs_model4_"))
    )
}


# 2) Standardize disease + model names
standardize_outcome_model <- function(df) {
  df %>%
    mutate(
      outcome_code = outcome,
      outcome = ifelse(outcome_code %in% names(outcome_label_map),
                       unname(outcome_label_map[outcome_code]),
                       outcome_code),
      model_type = ifelse(model_type %in% names(model_type_labels),
                     unname(model_type_labels[model_type]),
                     model_type)
    )
}


# 3) Run
dat_merge_res_std <- dat_merge_res %>%
  rename_metrics_cols() %>%
  standardize_outcome_model()

write_csv(dat_merge_res_std, "TableS3_model_performance.csv")




###### Table S4 全人群的描述性统计与主分析人群的描述性统计对比
### 连续型变量
dat_train_con_desc <- read_csv("Desc/sen_fullpopulation/train_sen_fullpopulation_desc_continuous_summary.csv")
dat_test_con_desc <- read_csv("Desc/sen_fullpopulation/test_sen_fullpopulation_desc_continuous_summary.csv")

dat_train_con_desc <- dat_train_con_desc[1:19,]
name_map <- c(
  BMI = "BMI (kg/m²)",
  waistcircumference = "Waist circumference (cm)",
  bpsystolic = "Systolic blood pressure (mmHg)",
  bpdiastolic = "Diastolic blood pressure (mmHg)",
  respiratoryrate = "Respiratory rate (breaths/min)",
  heartrate = "Heart rate (beats/min)",
  hemoglobin = "Hemoglobin (g/L)",
  Wbc = "White blood cell count (×10⁹/L)",
  platelet = "Platelet count (×10⁹/L)",
  fastingglucosemmol = "Fasting glucose (mmol/L)",
  ALT = "Alanine aminotransferase (U/L)",
  AST = "Aspartate aminotransferase (U/L)",
  totalbilirubin = "Total bilirubin (µmol/L)",
  creatinine = "Creatinine (µmol/L)",
  serumurea = "Urea (mmol/L)",
  totalcholesterol = "Total cholesterol (mmol/L)",
  triglycerides = "Triglycerides (mmol/L)",
  LDL = "LDL cholesterol (mmol/L)",
  HDL = "HDL cholesterol (mmol/L)"
)

dat_train_con_desc$variable_label <- name_map[dat_train_con_desc$variable]
dat_train_con_desc <- dat_train_con_desc %>%
  mutate(mean = case_when(
    variable=="bpsystolic" | variable=="bpdiastolic" | variable=="respiratoryrate" | variable=="heartrate" ~ round(mean,0),
    TRUE ~ round(mean, 2)
  )) %>%
  mutate(sd = case_when(
    variable=="bpsystolic" | variable=="bpdiastolic" | variable=="respiratoryrate" | variable=="heartrate" ~ round(sd,1),
    TRUE ~ round(sd, 2)
  ))

dat_train_con_desc <- dat_train_con_desc %>%
  mutate(desc=paste0(as.character(mean), " (±", as.character(sd), ")"))

write_csv(dat_train_con_desc, "train_sen_fullpopulation_desc.csv")


dat_test_con_desc$variable_label <- name_map[dat_test_con_desc$variable]
dat_test_con_desc <- dat_test_con_desc %>%
  mutate(mean = case_when(
    variable=="bpsystolic" | variable=="bpdiastolic" | variable=="respiratoryrate" | variable=="heartrate" ~ round(mean,0),
    TRUE ~ round(mean, 2)
  )) %>%
  mutate(sd = case_when(
    variable=="bpsystolic" | variable=="bpdiastolic" | variable=="respiratoryrate" | variable=="heartrate" ~ round(sd,1),
    TRUE ~ round(sd, 2)
  ))

dat_test_con_desc <- dat_test_con_desc %>%
  mutate(desc=paste0(as.character(mean), " (±", as.character(sd), ")"))

write_csv(dat_test_con_desc, "test_sen_fullpopulation_desc.csv")


### 疾病统计
dat_train_dis_desc <- read_csv("Desc/sen_fullpopulation/train_sen_fullpopulation_desc_disease4_summary.csv")
dat_test_dis_desc <- read_csv("Desc/sen_fullpopulation/test_sen_fullpopulation_desc_disease4_summary.csv")

dat_train_dis_desc$note <- NULL
dat_test_dis_desc$note <- NULL

## 20 种疾病
target_diseases <- c(
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
)

## 疾病名称规范映射
disease_label_map <- c(
  mi = "Myocardial infarction",
  afib_flutter = "Atrial fibrillation",
  cor_pulmonale = "Cor pulmonale",
  chf = "Heart failure",
  stroke = "All-cause stroke",
  ischemic_stroke = "Ischaemic stroke",
  hemorrhagic_stroke = "Haemorrhagic stroke",
  arterial_disease = "Arterial disease",
  copd = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure = "Liver failure",
  renal_failure = "Renal failure",
  diabetes = "Diabetes",
  thyroid_disease = "Thyroid disease",
  parkinson = "Parkinson's disease",
  dementia = "All-cause dementia",
  cancer_all = "All-cause cancer",
  liver_cancer = "Liver cancer",
  lung_cancer = "Lung cancer",
  kidney_cancer = "Kidney cancer"
)

## 描述性统计表
dat_dis_desc_final <- dat_train_dis_desc %>%
  # 只保留目标疾病
  filter(variable %in% target_diseases) %>%
  filter(str_detect(level, "^1:")) %>%
  # 规范疾病名称 + 加 incidence
  mutate(
    Disease = paste0(disease_label_map[variable], " incidence"),
    # 格式化 n (%)
    `Cases, n (%)` = paste0(
      comma(n),
      " (",
      formatC(pct, format = "f", digits = 2),
      "%)"
    )
  ) %>%
  select(Disease, `Cases, n (%)`) %>%
  arrange(match(names(disease_label_map), Disease)) %>% 
  distinct()

write_csv(dat_dis_desc_final, "train_sen_fullpopulation_dis_desc.csv")

# test
dat_dis_desc_final <- dat_test_dis_desc %>%
  # 只保留目标疾病
  filter(variable %in% target_diseases) %>%
  filter(str_detect(level, "^1:")) %>%
  # 规范疾病名称 + 加 incidence
  mutate(
    Disease = paste0(disease_label_map[variable], " incidence"),
    # 格式化 n (%)
    `Cases, n (%)` = paste0(
      comma(n),
      " (",
      formatC(pct, format = "f", digits = 2),
      "%)"
    )
  ) %>%
  select(Disease, `Cases, n (%)`) %>%
  arrange(match(names(disease_label_map), Disease)) %>% 
  distinct()

write_csv(dat_dis_desc_final, "test_sen_fullpopulation_dis_desc.csv")








###### Table S5 全人群上模型的各项评价指标（嵌套交叉验证、地理外部验证结果）
dat_fullpopulation_res <- read_csv("XGBoost/sen_fullpopulation/summary_metrics.csv")

dat_merge_res <- dat_fullpopulation_res
dat_merge_res <- select(dat_merge_res, -c(23:25,46:48))
dat_merge_res$model <- "XGBoost"
dat_merge_res <- select(dat_merge_res, 1,43,2:42)

# 0) Standard name maps
# Disease standard names
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)

# Model standard names (5)
model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

# Prefixes you requested
cv_prefix   <- "nested_cross_validation_"
test_prefix <- "geographic_external_val_"


# 1) Rename columns: cv_ -> nested_cross_validation_ ; test_ -> geographic_external_val_
rename_metrics_cols <- function(df) {
  df %>%
    
    rename_with(~ str_replace_all(.x, "^cv_",   cv_prefix),   .cols = matches("^cv_")) %>%
    rename_with(~ str_replace_all(.x, "^test_", test_prefix), .cols = matches("^test_")) %>%
    
    
    rename_with(
      ~ str_replace_all(.x,
                        paste0("^", cv_prefix, "vs_model5_"),
                        paste0(cv_prefix, "vs_model5_")),
      .cols = matches(paste0("^", cv_prefix, "vs_model5_"))
    ) %>%
    
    
    rename_with(
      ~ str_replace_all(.x,
                        paste0("^", cv_prefix, "model5_vs_model4_"),
                        paste0(cv_prefix, "model5_vs_model4_")),
      .cols = matches(paste0("^", cv_prefix, "model5_vs_model4_"))
    ) %>%
    
    
    rename_with(
      ~ str_replace_all(.x,
                        paste0("^", test_prefix, "vs_model5_"),
                        paste0(test_prefix, "vs_model5_")),
      .cols = matches(paste0("^", test_prefix, "vs_model5_"))
    ) %>%
    
    
    rename_with(
      ~ str_replace_all(.x,
                        paste0("^", test_prefix, "model5_vs_model4_"),
                        paste0(test_prefix, "model5_vs_model4_")),
      .cols = matches(paste0("^", test_prefix, "model5_vs_model4_"))
    )
}


# 2) Standardize disease + model names
standardize_outcome_model <- function(df) {
  df %>%
    mutate(
      outcome_code = outcome,
      outcome = ifelse(outcome_code %in% names(outcome_label_map),
                       unname(outcome_label_map[outcome_code]),
                       outcome_code),
      model_type = ifelse(model_type %in% names(model_type_labels),
                          unname(model_type_labels[model_type]),
                          model_type)
    )
}


# 3) Run
dat_merge_res_std <- dat_merge_res %>%
  rename_metrics_cols() %>%
  standardize_outcome_model()

write_csv(dat_merge_res_std, "TableS5_fullpopulation_val.csv")






###### Table S6 主模型临床获益的DR结果（主人群、全人群）
dat_main_DR <- read_csv("XGBoost/DR_curve/dr_curve_fpr_grid.csv")
dat_fullpopulation_DR <- read_csv("XGBoost/DR_curve/dr_curve_fpr_grid_sen_fullpopulation.csv")

dat_main_DR$note <- "Main analysis"
dat_fullpopulation_DR$note <- "Full population analysis"

dat_DR_merge <- rbind(dat_main_DR, dat_fullpopulation_DR)

dat_DR_merge <- select(dat_DR_merge, 2:5,7)


# 1) 映射：疾病标准名称
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)


# 2) 映射：模型标准名称
model_label_map <- c(
  model2_clinical               = "Model 2: Clinical",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)


# 3) 标准化 outcome & model
dat_DR_merge_std <- dat_DR_merge %>%
  mutate(
    outcome_code = outcome,
    model_code   = model,
    
    outcome = ifelse(
      outcome_code %in% names(outcome_label_map),
      unname(outcome_label_map[outcome_code]),
      outcome_code
    ),
    
    model = ifelse(
      model_code %in% names(model_label_map),
      unname(model_label_map[model_code]),
      model_code
    )
  )

dat_DR_merge_std <- select(dat_DR_merge_std,1:5)
names(dat_DR_merge_std)[2] <- "model_type"
write_csv(dat_DR_merge_std, "TableS6_DR_res.csv")


###### Table S7 主模型临床获益的LR结果（主人群、全人群）
dat_main_LR <- read_csv("XGBoost/LR_curve/LR_curve_fpr_grid.csv")
dat_fullpopulation_LR <- read_csv("XGBoost/LR_curve/LR_curve_fpr_grid_sen_fullpopulation.csv")

dat_main_LR$note <- "Main analysis"
dat_fullpopulation_LR$note <- "Full population analysis"

dat_LR_merge <- rbind(dat_main_LR, dat_fullpopulation_LR)

dat_LR_merge <- select(dat_LR_merge, 2:4,8,10)
names(dat_LR_merge)[c(3,4)] <- c("FPR_grid", "LR")


# 1) 映射：疾病标准名称
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)


# 2) 映射：模型标准名称
model_label_map <- c(
  model2_clinical               = "Model 2: Clinical",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)


# 3) 标准化 outcome & model
dat_LR_merge_std <- dat_LR_merge %>%
  mutate(
    outcome_code = outcome,
    model_code   = model,
    
    outcome = ifelse(
      outcome_code %in% names(outcome_label_map),
      unname(outcome_label_map[outcome_code]),
      outcome_code
    ),
    
    model = ifelse(
      model_code %in% names(model_label_map),
      unname(model_label_map[model_code]),
      model_code
    )
  )

dat_LR_merge_std <- select(dat_LR_merge_std,1:5)
names(dat_LR_merge_std)[2] <- "model_type"
write_csv(dat_LR_merge_std, "TableS7_LR_res.csv")




###### Table S8 UKB人群的具体结果
dat_ukb_res <- read_csv("XGBoost/all_models_summary_ukb_validation_year_10_pvalue.csv")

# 1) 标准疾病名称映射
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)


# 2) 标准模型名称映射
model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)


# 3) 标准化 UKB 结果表
dat_ukb_res_std <- dat_ukb_res %>%
  mutate(
    outcome_code = outcome,
    model_type_code = model_type,
    
    # 标准化疾病名
    outcome = ifelse(
      outcome_code %in% names(outcome_label_map),
      unname(outcome_label_map[outcome_code]),
      outcome_code
    ),
    
    # 标准化模型名（生成新列，不覆盖 model_type）
    model = ifelse(
      model_type_code %in% names(model_type_labels),
      unname(model_type_labels[model_type_code]),
      model_type_code
    )
  )

dat_ukb_res_std$model_type <- dat_ukb_res_std$model
dat_ukb_res_std$model <- "XGBoost"

dat_ukb_res_std <- select(dat_ukb_res_std, 1,17,2:14)

write_csv(dat_ukb_res_std, "TableS8_ukb_res.csv")




###### Table S9 亚组的AUROC结果
dat_subgroup_roc <- read_csv("XGBoost/subgroup_analysis/subgroup_performance.csv")
dat_subgroup_p <- read_csv("XGBoost/subgroup_analysis/subgroup_roc_pr_diff.csv")

dat_subgroup_roc <- subset(dat_subgroup_roc, subgroup_var!="city_raw")
dat_subgroup_p <- subset(dat_subgroup_p, subgroup_var!="city_raw")

dat_subgroup_roc$n <- NULL
dat_subgroup_roc$n_events <- NULL

dat_subgroup_p <- select(dat_subgroup_p, 1,2,3,7,11)
dat_subgroup_p$model_type <- "model4_clinical_baseline_exam"

names(dat_subgroup_p)[c(4,5)] <- c("model4_vs_model5_auroc_pvalue", "model4_vs_model5_auprc_pvalue")
dat_subgroup_p$subgroup_var <- NULL

dat_subgroup_merge <- dat_subgroup_roc %>%
  left_join(dat_subgroup_p, by = c("outcome", "model_type", "subgroup_level"))


# 1) 标准疾病名称映射
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)


# 2) 标准模型名称映射
model_type_labels <- c(
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)


# 3) 标准亚组变量/水平映射
subgroup_var_labels <- c(
  age_group    = "Age",
  sex          = "Sex",
  hypertension = "Hypertension"
)

subgroup_level_labels <- c(
  old      = ">=60 years",
  young_60 = "<60 years",
  male     = "Male",
  female   = "Female",
  `1.0`    = "Yes",
  `0.0`    = "No"
)


# 4) 应用标准化
dat_subgroup_merge_std <- dat_subgroup_merge %>%
  mutate(
    outcome_code        = outcome,
    model_type_code     = model_type,
    subgroup_var_code   = subgroup_var,
    subgroup_level_code = subgroup_level,
    
    outcome = ifelse(
      outcome_code %in% names(outcome_label_map),
      unname(outcome_label_map[outcome_code]),
      outcome_code
    ),
    
    model = ifelse(
      model_type_code %in% names(model_type_labels),
      unname(model_type_labels[model_type_code]),
      model_type_code
    ),
    
    subgroup_var_label = ifelse(
      subgroup_var_code %in% names(subgroup_var_labels),
      unname(subgroup_var_labels[subgroup_var_code]),
      subgroup_var_code
    ),
    
    subgroup_level_label = ifelse(
      subgroup_level_code %in% names(subgroup_level_labels),
      unname(subgroup_level_labels[subgroup_level_code]),
      subgroup_level_code
    ),
    
    subgroup = paste0(subgroup_var_label, ": ", subgroup_level_label)
  )

dat_subgroup_merge_std <- select(dat_subgroup_merge_std, 1,23,26,5:18)
names(dat_subgroup_merge_std)[2] <- "model_type"

write_csv(dat_subgroup_merge_std, "TableS9_subgroup_roc.csv")



###### Table S10 亚组的NRI结果
dat_subgroup_nri <- read_csv("XGBoost/subgroup_analysis/subgroup_nri_idi.csv")
dat_subgroup_nri <- subset(dat_subgroup_nri, subgroup_var!="city_raw")

dat_subgroup_nri <- select(dat_subgroup_nri,1:14)
dat_subgroup_nri$n <- NULL
dat_subgroup_nri$n_events <- NULL

names(dat_subgroup_nri)[4] <- "nri"


# 1) 标准疾病名称映射
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)


# 2) 标准亚组变量/水平映射
subgroup_var_labels <- c(
  age_group    = "Age",
  sex          = "Sex",
  hypertension = "Hypertension"
)

subgroup_level_labels <- c(
  old      = ">=60 years",
  young_60 = "<60 years",
  male     = "Male",
  female   = "Female",
  `1.0`    = "Yes",
  `0.0`    = "No"
)


# 3) 应用标准化
dat_subgroup_nri_std <- dat_subgroup_nri %>%
  mutate(
    outcome_code        = outcome,
    subgroup_var_code   = subgroup_var,
    subgroup_level_code = subgroup_level,
    
    outcome = ifelse(
      outcome_code %in% names(outcome_label_map),
      unname(outcome_label_map[outcome_code]),
      outcome_code
    ),
    
    subgroup_var_label = ifelse(
      subgroup_var_code %in% names(subgroup_var_labels),
      unname(subgroup_var_labels[subgroup_var_code]),
      subgroup_var_code
    ),
    
    subgroup_level_label = ifelse(
      subgroup_level_code %in% names(subgroup_level_labels),
      unname(subgroup_level_labels[subgroup_level_code]),
      subgroup_level_code
    ),
    
    subgroup = paste0(subgroup_var_label, ": ", subgroup_level_label)
  )

dat_subgroup_nri_std <- select(dat_subgroup_nri_std, 1,18,4:12)
dat_subgroup_nri_std$note <- "model5_vs_model4_NRI"
dat_subgroup_nri_std <- select(dat_subgroup_nri_std, 1,2,12,3:11)

write_csv(dat_subgroup_nri_std, "TableS10_subgroup_nri.csv")




###### Table S11 敏感性分析不同预测时间窗口的模型结果
dat_sen_pred_1yr <- read_csv("XGBoost/sen_timewindows/all_models_summary_sen_1yr.csv")
dat_sen_pred_2yr <- read_csv("XGBoost/sen_timewindows/all_models_summary_sen_2yr.csv")
dat_sen_pred_3yr <- read_csv("XGBoost/sen_timewindows/all_models_summary_sen_3yr.csv")

dat_sen_pred_1yr$prediction_timewindow <- "1-year"
dat_sen_pred_2yr$prediction_timewindow <- "2-year"
dat_sen_pred_3yr$prediction_timewindow <- "3-year"

dat_sen_pred_merge <- rbind(dat_sen_pred_1yr,
                            dat_sen_pred_2yr,
                            dat_sen_pred_3yr)

dat_sen_pred_merge <- select(dat_sen_pred_merge, 1,2,13,3:11)


# 1) 映射：疾病标准名称
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)


# 2) 映射：模型标准名称
model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)


# 3) 改列名：ext_ -> geographic_external_val_
dat_sen_pred_merge_std <- dat_sen_pred_merge

names(dat_sen_pred_merge_std) <- names(dat_sen_pred_merge_std) %>%
  str_replace("^ext_", "geographic_external_val_")


# 4) 标准化 outcome & model_type
dat_sen_pred_merge_std <- dat_sen_pred_merge_std %>%
  mutate(
    outcome_code = outcome,
    model_type_code = model_type,
    
    outcome = ifelse(
      outcome_code %in% names(outcome_label_map),
      unname(outcome_label_map[outcome_code]),
      outcome_code
    ),
    
    model_type = ifelse(
      model_type_code %in% names(model_type_labels),
      unname(model_type_labels[model_type_code]),
      model_type_code
    )
  )

dat_sen_pred_merge_std <- select(dat_sen_pred_merge_std, 1:12)
write_csv(dat_sen_pred_merge_std, "TableS11_geo_pred_timewindow.csv")



###### Table S12 敏感性分析排除基线后不同时间段发病人群的模型结果
dat_sen_exclude_1mon <- read_csv("XGBoost/sen_timewindows/all_models_summary_sen_exclude_1mon.csv")
dat_sen_exclude_3mon <- read_csv("XGBoost/sen_timewindows/all_models_summary_sen_exclude_3mon.csv")
dat_sen_exclude_6mon <- read_csv("XGBoost/sen_timewindows/all_models_summary_sen_exclude_6mon.csv")
dat_sen_exclude_12mon <- read_csv("XGBoost/sen_timewindows/all_models_summary_sen_exclude_12mon.csv")

dat_sen_exclude_1mon$exclude_timewindow <- "1-month"
dat_sen_exclude_3mon$exclude_timewindow <- "3-month"
dat_sen_exclude_6mon$exclude_timewindow <- "6-month"
dat_sen_exclude_12mon$exclude_timewindow <- "12-month"

dat_sen_exclude_merge <- rbind(dat_sen_exclude_1mon,
                               dat_sen_exclude_3mon,
                               dat_sen_exclude_6mon,
                               dat_sen_exclude_12mon)

dat_sen_exclude_merge <- select(dat_sen_exclude_merge, 1,2,13,3:11)

# 1) 映射：疾病标准名称
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)


# 2) 映射：模型标准名称
model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)


# 3) 改列名：ext_ -> geographic_external_val_
dat_sen_exclude_merge_std <- dat_sen_exclude_merge

names(dat_sen_exclude_merge_std) <- names(dat_sen_exclude_merge_std) %>%
  str_replace("^ext_", "geographic_external_val_")


# 4) 标准化 outcome & model_type
dat_sen_exclude_merge_std <- dat_sen_exclude_merge_std %>%
  mutate(
    outcome_code = outcome,
    model_type_code = model_type,
    
    outcome = ifelse(
      outcome_code %in% names(outcome_label_map),
      unname(outcome_label_map[outcome_code]),
      outcome_code
    ),
    
    model_type = ifelse(
      model_type_code %in% names(model_type_labels),
      unname(model_type_labels[model_type_code]),
      model_type_code
    )
  )

dat_sen_exclude_merge_std <- select(dat_sen_exclude_merge_std, 1:12)
write_csv(dat_sen_exclude_merge_std, "TableS12_geo_exclude_timewindow.csv")



###### Table S13, icd10
suppressPackageStartupMessages({
  library(dplyr)
  library(purrr)
  library(tibble)
})

disease_codes <- list(
  # 1. 心血管疾病
  "重大心血管不良事件" = c("G45", "I21", "I22", "I23", "I24", "I25", "I26", "I30", "I33", "I38", "I40", "I46", "I50", "I60", "I61", "I62", "I63", "I64", "I71", "I74"),
  "冠心病" = c("I20", "I21", "I22", "I23", "I24", "I25"),
  "心肌梗死" = c("I21", "I22"),
  "肺心病" = c("I26", "I27", "I28"),
  "心脏瓣膜疾病" = c("I33", "I34", "I35", "I36", "I37", "I38", "I39"),
  "心肌病" = c("I42"),
  "传导阻滞" = c("I44", "I45"),
  "心房颤动和扑动" = c("I48"),
  "心力衰竭" = c("I50"),
  "动脉疾病" = c("I70", "I71", "I72", "I73", "I74", "I77", "I78"),
  
  # 2. 脑血管疾病
  "中风" = c("I60", "I61", "I62", "I63", "I64"),
  "缺血性中风" = c("I63"),
  "出血性中风" = c("I60", "I61", "I62"),
  
  # 3. 呼吸系统疾病
  "慢性阻塞性肺疾病" = c("J40", "J41", "J42", "J43", "J44", "J47"),
  "肺气肿" = c("J43"),
  "哮喘" = c("J45", "J46"),
  
  # 4. 消化系统疾病
  "肝纤维化和肝硬化" = c("K74"),
  "肝衰竭" = c("K72"),
  "胆结石" = c("K80"),
  "胆囊炎" = c("K81"),
  
  # 5. 泌尿系统疾病
  "肾炎" = c("N00", "N01", "N02", "N03", "N04", "N05", "N10", "N11", "N12"),
  "肾结石" = c("N20"),
  "肾衰竭" = c("N17", "N18", "N19"),
  
  # 6. 代谢系统疾病
  "糖尿病" = c("E10", "E11", "E12", "E13", "E14"),
  "甲状腺疾病" = c("E01", "E02", "E03", "E04", "E05", "E06", "E07"),
  
  # 7. 神经系统疾病
  "帕金森症" = c("G20", "G21", "G22"),
  "全因痴呆症" = c("F00", "F01", "F02", "F03", "G30", "G31"),
  "癫痫" = c("G40", "G41"),
  
  # 8. 恶性肿瘤
  "泛癌" = c("C00","C01","C02","C03","C04","C05","C06","C09","C10","C11","C12","C13","C07","C08","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26","C32","C33","C34","C30","C31","C37","C38","C39","C40","C43","C44","C45","C46","C47","C48","C49","C50","C53","C54","C56","C55","C51","C52","C57","C58","C60","C61","C62","C63","C64","C65","C66","C68","C67","C69","C70","C71","C72","C73","C74","C75","C76","C80","C77","C78","C79","C81","C82","C83","C85","C84","C86","C96","C88","C90","C91","C92","C93","C94","C95"),
  "肝癌" = c("C22"),
  "胆囊癌" = c("C23"),
  "胰腺癌" = c("C25"),
  "结直肠癌" = c("C18", "C19", "C20"),
  "肺癌" = c("C33", "C34"),
  "肾癌" = c("C64", "C65", "C66", "C68"),
  "淋巴癌" = c("C81", "C82", "C83", "C85", "C84", "C86", "C96"),
  
  # 9.作为纳排疾病或疾病史
  "高血压" = c("I10", "I15"),
  "高血压性心脏病" = c("I11"),
  "心肌炎" = c("I40", "I41"),
  "肺炎" = c("J12", "J13", "J14", "J15", "J16", "J17", "J18"),
  "病毒性肝炎" = c("B15", "B16", "B17", "B18", "B19"),
  "慢性肝炎" = c("K73", "K75"),
  "酒精性肝病" = c("K70"),
  "脂肪肝" = c("K76"),
  "胰腺炎" = c("K85", "K86"),
  "高脂血症" = c("E78.5"),
  "抑郁症" = c("F32", "F33")
)

outcome_label_map_20 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)

outcome_cn_map_20 <- c(
  mi                       = "心肌梗死",
  afib_flutter             = "心房颤动和扑动",
  cor_pulmonale            = "肺心病",
  chf                      = "心力衰竭",
  stroke                   = "中风",
  ischemic_stroke          = "缺血性中风",
  hemorrhagic_stroke       = "出血性中风",
  arterial_disease         = "动脉疾病",
  copd                     = "慢性阻塞性肺疾病",
  liver_fibrosis_cirrhosis = "肝纤维化和肝硬化",
  liver_failure            = "肝衰竭",
  renal_failure            = "肾衰竭",
  diabetes                 = "糖尿病",
  thyroid_disease          = "甲状腺疾病",
  parkinson                = "帕金森症",
  dementia                 = "全因痴呆症",
  cancer_all               = "泛癌",
  liver_cancer             = "肝癌",
  lung_cancer              = "肺癌",
  kidney_cancer            = "肾癌"
)

icd10_table_20 <- tibble(
  outcome_code = names(outcome_label_map_20),         # e.g., "mi"
  outcome      = unname(outcome_label_map_20),        # standard outcome name you want
  cn_key       = unname(outcome_cn_map_20)            # Chinese name used to index disease_codes
) %>%
  mutate(
    # fetch ICD10 code vector from disease_codes by cn_key
    icd10_vec = map(cn_key, function(k) {
      if (!is.character(k) || length(k) != 1 || is.na(k)) return(NULL)
      if (!k %in% names(disease_codes)) return(NULL)
      v <- disease_codes[[k]]
      if (is.null(v)) return(NULL)
      as.character(v)
    }),
    icd10 = map_chr(icd10_vec, function(v) {
      if (is.null(v) || length(v) == 0) return(NA_character_)
      paste(v, collapse = ", ")
    })
  ) %>%
  select(outcome, icd10)

write_csv(icd10_table_20, "TableS13_icd10.csv")




###### Figure S1 CatBoost模型校准曲线
###### 校准曲线
# 0) Paths
base_dir   <- "CatBoost"
pred_dir   <- file.path(base_dir, "outer_cv_predictions")

# 1) Outcomes 
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)
outcomes_20  <- names(outcome_label_map)
panel_levels <- unname(outcome_label_map)

# 2) Models (5) 
model_types_5 <- c(
  "model1_base",
  "model2_clinical",
  "model3_dynamic",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

# 3) Colors
model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3",
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)

# 4) Read prediction file
read_pred_one <- function(outcome, model_type) {
  f <- file.path(pred_dir, sprintf("%s_%s_outer5fold_predictions.csv", outcome, model_type))
  if (!file.exists(f)) {
    warning("Missing prediction file: ", f)
    return(NULL)
  }
  readr::read_csv(f, show_col_types = FALSE) %>%
    transmute(
      outcome = outcome,
      model_type = model_type,
      actual = as.numeric(actual),
      pred   = as.numeric(pred_raw)
    ) %>%
    filter(!is.na(actual), !is.na(pred)) %>%
    mutate(pred = pmin(pmax(pred, 0), 1))
}

pred_df <- bind_rows(lapply(outcomes_20, function(oc) {
  bind_rows(lapply(model_types_5, function(mt) read_pred_one(oc, mt)))
}))

# 5) Quantile-binned calibration lines
get_calib_bins <- function(df_one, n_bins = 20) {
  if (nrow(df_one) < n_bins) return(NULL)
  
  df_one %>%
    mutate(bin = ntile(pred, n_bins)) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(pred) * 100,
      obs_rate  = mean(actual) * 100,
      .groups = "drop"
    ) %>%
    arrange(mean_pred)
}

n_bins <- 20

calib_df <- pred_df %>%
  group_by(outcome, model_type) %>%
  group_modify(~{
    out <- get_calib_bins(.x, n_bins = n_bins)
    if (is.null(out)) return(tibble())
    out
  }) %>%
  ungroup() %>%
  mutate(
    outcome_label = factor(outcome_label_map[outcome], levels = panel_levels),
    model = factor(model_type_labels[model_type],
                   levels = unname(model_type_labels[model_types_5]))
  )

# 6) Better per-panel axis maxima
q_focus <- 0.99          
min_cap <- 0.25          
max_cap <- 100

panel_limits <- calib_df %>%
  group_by(outcome_label) %>%
  summarise(
    x_q = suppressWarnings(quantile(mean_pred, probs = q_focus, na.rm = TRUE)),
    y_q = suppressWarnings(quantile(obs_rate,  probs = q_focus, na.rm = TRUE)),
    x_max = max(mean_pred, na.rm = TRUE),
    y_max = max(obs_rate,  na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    m_q = pmax(ifelse(is.finite(x_q), x_q, x_max),
               ifelse(is.finite(y_q), y_q, y_max),
               na.rm = TRUE),
    
    m_max = m_q + pmax(0.15 * m_q, 0.05),
    
    m_max = pmin(pmax(m_max, min_cap), max_cap)
  )

# 7) Panel-specific breaks/labels
make_breaks <- function(maxv) {
  # Choose step based on maxv
  step <- if (maxv <= 0.5) 0.1 else if (maxv <= 1) 0.2 else if (maxv <= 2) 0.5 else if (maxv <= 10) 2 else if (maxv <= 25) 5 else 10
  seq(0, maxv, by = step)
}

make_labels <- function(maxv) {
  # show one decimal for very small ranges
  if (maxv <= 1) label_number(accuracy = 0.1) else label_number(accuracy = 1)
}

x_scales <- lapply(panel_levels, function(lbl) {
  maxv <- panel_limits$m_max[match(lbl, panel_limits$outcome_label)]
  scale_x_continuous(
    limits = c(0, maxv),
    breaks = make_breaks(maxv),
    labels = make_labels(maxv),
    expand = expansion(mult = c(0.02, 0.04))
  )
})

y_scales <- lapply(panel_levels, function(lbl) {
  maxv <- panel_limits$m_max[match(lbl, panel_limits$outcome_label)]
  scale_y_continuous(
    limits = c(0, maxv),
    breaks = make_breaks(maxv),
    labels = make_labels(maxv),
    expand = expansion(mult = c(0.02, 0.04))
  )
})

# 8) Theme
theme_calib <- theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14, color = "#1F2328", hjust = 0.5),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10, color = "#1F2328"),
    
    axis.title.x = element_text(
      size = 11,
      face = "bold",        
      color = "#1F2328",
      margin = margin(t = 6)
    ),
    axis.title.y = element_text(
      size = 11,
      face = "bold",       
      color = "#1F2328",
      margin = margin(r = 6)
    ),
    axis.text = element_text(size = 9, color = "#1F2328"),
    axis.ticks = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line  = element_line(linewidth = 0.35, color = "#2B2B2B"),
    
    panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
    panel.grid.major.x = element_line(color = "#F5F7FA", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 9.5, color = "#1F2328"),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    
    plot.margin = margin(10, 12, 8, 10)
  )

# 9) Plot
p_calib <- ggplot(calib_df, aes(x = mean_pred, y = obs_rate, color = model, group = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.35, color = "#8B949E") +
  geom_line(linewidth = 0.85, alpha = 0.95) +
  scale_color_manual(values = model_cols) +
  labs(
    title = "Calibration Curve",
    x = "Predicted risk (%)",
    y = "Observed event rate (%)"
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_calib +
  ggh4x::facet_wrap2(~ outcome_label, nrow = 4, ncol = 5, scales = "free", axes = "all") +
  ggh4x::facetted_pos_scales(x = x_scales, y = y_scales)

p_calib

ggsave("FigS1_catboost_calibration.pdf", p_calib, width = 10, height = 7)



###### Figure S2 CatBoost模型DCA结果
###### DCA
suppressPackageStartupMessages({
  library(tidyverse)
  library(glue)
  library(scales)
})

# 0) Paths
base_dir   <- "CatBoost"
dca_dir    <- file.path(base_dir, "dca_outputs", "cv")

# 1) Outcomes (12) + labels
outcome_label_map_12 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease"
)
outcomes_12  <- names(outcome_label_map_12)
panel_levels <- unname(outcome_label_map_12)

# 2) Models (4) + labels
model_types_4 <- c(
  "model1_base",
  "model2_clinical",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)
model_levels_4 <- unname(model_type_labels[model_types_4])

# 3) Colors
model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3", # not used
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)

fill_color <- scales::alpha(model_cols[["Model 5: Clin+Exam (longitudinal)"]], 0.25)

# 4) Read DCA CSVs
read_dca_one <- function(outcome) {
  f <- file.path(dca_dir, glue("dca_{outcome}.csv"))
  if (!file.exists(f)) {
    warning("Missing DCA file: ", f)
    return(NULL)
  }
  
  readr::read_csv(f, show_col_types = FALSE) %>%
    filter(model_type %in% c(model_types_4, "treat_all", "treat_none")) %>%
    transmute(
      outcome       = outcome,
      model_type    = as.character(model_type),
      thresholds    = as.numeric(thresholds) * 100,  # -> %
      NB            = as.numeric(NB),
      sNB           = as.numeric(sNB) * 100,          # -> %
      prevalence    = as.numeric(prevalence),
      outcome_label = factor(outcome_label_map_12[outcome], levels = panel_levels)
    ) %>%
    filter(!is.na(thresholds), !is.na(sNB))
}

dca_df_raw <- bind_rows(lapply(outcomes_12, read_dca_one))

# 5) Keep model and reference data, add labels
dca_models_raw <- dca_df_raw %>%
  filter(model_type %in% model_types_4) %>%
  mutate(
    model = model_type_labels[model_type],
    model = factor(model, levels = model_levels_4)
  )

dca_ref <- dca_df_raw %>%
  filter(model_type %in% c("treat_all", "treat_none")) %>%
  group_by(outcome_label, model_type) %>%
  arrange(thresholds) %>%
  ungroup()

# 6) Global threshold truncation
model5_type <- "model5_clinical_dynamic_exam"

thr_limits <- dca_models_raw %>%
  filter(model_type == model5_type) %>%
  group_by(outcome_label) %>%
  arrange(thresholds) %>%
  summarise(
    thr_cut = suppressWarnings(min(thresholds[sNB <= 0], na.rm = TRUE)),
    thr_max_data = suppressWarnings(max(thresholds, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(
    thr_cut = ifelse(is.finite(thr_cut), thr_cut, thr_max_data),
    thr_max = pmin(thr_cut + 2, thr_max_data),
    thr_max = pmax(thr_max, 5)
  )

dca_models_raw <- dca_models_raw %>%
  left_join(thr_limits, by = "outcome_label") %>%
  filter(thresholds <= thr_max) %>%
  select(-thr_cut, -thr_max_data, -thr_max)

dca_ref <- dca_ref %>%
  left_join(thr_limits, by = "outcome_label") %>%
  filter(thresholds <= thr_max) %>%
  select(-thr_cut, -thr_max_data, -thr_max)

# 7) Enforce "floor at 0" and "absorbing at 0" for model curves
# Rule:
#   - any sNB < 0 -> 0
#   - once sNB reaches 0 (after flooring), keep sNB=0 for all later thresholds
enforce_floor_absorb0 <- function(df) {
  df <- df %>% arrange(thresholds)
  s <- pmax(df$sNB, 0)
  first_zero <- which(s <= 0)[1]
  if (!is.na(first_zero)) {
    s[first_zero:length(s)] <- 0
  }
  df$sNB <- s
  df
}

dca_models <- dca_models_raw %>%
  group_by(outcome_label, model_type) %>%
  group_modify(~ enforce_floor_absorb0(.x)) %>%
  ungroup() %>%
  mutate(
    # keep ordering stable
    model = factor(model_type_labels[model_type], levels = model_levels_4)
  )

# 8) Ribbon between Model 4 and Model 5 (use adjusted sNB)
dca_ribbon <- dca_models %>%
  filter(model %in% c("Model 4: Clin+Exam (baseline)",
                      "Model 5: Clin+Exam (longitudinal)")) %>%
  select(outcome_label, thresholds, model, sNB) %>%
  pivot_wider(names_from = model, values_from = sNB) %>%
  drop_na(`Model 4: Clin+Exam (baseline)`, `Model 5: Clin+Exam (longitudinal)`) %>%
  arrange(outcome_label, thresholds) %>%
  mutate(
    ymin = pmin(`Model 4: Clin+Exam (baseline)`, `Model 5: Clin+Exam (longitudinal)`),
    ymax = pmax(`Model 4: Clin+Exam (baseline)`, `Model 5: Clin+Exam (longitudinal)`)
  )

# 9) Theme
theme_dca <- theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14, color = "#1F2328", hjust = 0.5),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10, color = "#1F2328"),
    
    axis.title.x = element_text(size = 11, face = "bold", color = "#1F2328", margin = margin(t = 6)),
    axis.title.y = element_text(size = 11, face = "bold", color = "#1F2328", margin = margin(r = 6)),
    axis.text = element_text(size = 9, color = "#1F2328"),
    
    axis.ticks = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line  = element_line(linewidth = 0.45, color = "#2B2B2B"),
    
    panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
    panel.grid.major.x = element_line(color = "#F5F7FA", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 9.5, color = "#1F2328"),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    
    plot.margin = margin(10, 12, 8, 10)
  )

# 10) Plot
treat_all_df  <- dca_ref %>% filter(model_type == "treat_all")
treat_none_df <- dca_ref %>% filter(model_type == "treat_none")

p_dca <- ggplot() +
  # Treat-none
  geom_line(
    data = treat_none_df,
    aes(x = thresholds, y = sNB, group = outcome_label),
    color = "#6B7280",
    linewidth = 0.45,
    linetype = "solid",
    alpha = 0.95
  ) +
  # Treat-all
  geom_line(
    data = treat_all_df,
    aes(x = thresholds, y = sNB, group = outcome_label),
    color = "#9CA3AF",
    linewidth = 0.45,
    linetype = "solid",
    alpha = 0.95
  ) +
  # Ribbon between Model 4 and Model 5
  geom_ribbon(
    data = dca_ribbon,
    aes(x = thresholds, ymin = ymin, ymax = ymax, group = outcome_label),
    fill = fill_color
  ) +
  # Model curves (ALL SOLID)
  geom_line(
    data = dca_models,
    aes(x = thresholds, y = sNB, color = model, group = interaction(outcome_label, model)),
    linewidth = 0.7,
    alpha = 0.95,
    linetype = "solid"
  ) +
  facet_wrap(~ outcome_label, nrow = 3, ncol = 4, scales = "free_x") +
  scale_color_manual(values = model_cols[model_levels_4]) +
  coord_cartesian(ylim = c(-1, NA)) +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.04))) +
  labs(
    x = "Threshold probability (%)",
    y = "Standardized net benefit (%)"
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_dca

p_dca

ggsave("FigS2_catboost_DCA.pdf", p_dca, width = 10, height = 7)





###### Figure S3 LightGBM模型校准曲线
###### 校准曲线
# 0) Paths
base_dir   <- "LightGBM"
pred_dir   <- file.path(base_dir, "outer_cv_predictions")

# 1) Outcomes 
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)
outcomes_20  <- names(outcome_label_map)
panel_levels <- unname(outcome_label_map)

# 2) Models (5) 
model_types_5 <- c(
  "model1_base",
  "model2_clinical",
  "model3_dynamic",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

# 3) Colors
model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3",
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)

# 4) Read prediction file
read_pred_one <- function(outcome, model_type) {
  f <- file.path(pred_dir, sprintf("%s_%s_outer5fold_predictions.csv", outcome, model_type))
  if (!file.exists(f)) {
    warning("Missing prediction file: ", f)
    return(NULL)
  }
  readr::read_csv(f, show_col_types = FALSE) %>%
    transmute(
      outcome = outcome,
      model_type = model_type,
      actual = as.numeric(actual),
      pred   = as.numeric(pred_raw)
    ) %>%
    filter(!is.na(actual), !is.na(pred)) %>%
    mutate(pred = pmin(pmax(pred, 0), 1))
}

pred_df <- bind_rows(lapply(outcomes_20, function(oc) {
  bind_rows(lapply(model_types_5, function(mt) read_pred_one(oc, mt)))
}))

# 5) Quantile-binned calibration lines
get_calib_bins <- function(df_one, n_bins = 20) {
  if (nrow(df_one) < n_bins) return(NULL)
  
  df_one %>%
    mutate(bin = ntile(pred, n_bins)) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(pred) * 100,
      obs_rate  = mean(actual) * 100,
      .groups = "drop"
    ) %>%
    arrange(mean_pred)
}

n_bins <- 20

calib_df <- pred_df %>%
  group_by(outcome, model_type) %>%
  group_modify(~{
    out <- get_calib_bins(.x, n_bins = n_bins)
    if (is.null(out)) return(tibble())
    out
  }) %>%
  ungroup() %>%
  mutate(
    outcome_label = factor(outcome_label_map[outcome], levels = panel_levels),
    model = factor(model_type_labels[model_type],
                   levels = unname(model_type_labels[model_types_5]))
  )

# 6) Better per-panel axis maxima
q_focus <- 0.99          
min_cap <- 0.25          
max_cap <- 100

panel_limits <- calib_df %>%
  group_by(outcome_label) %>%
  summarise(
    x_q = suppressWarnings(quantile(mean_pred, probs = q_focus, na.rm = TRUE)),
    y_q = suppressWarnings(quantile(obs_rate,  probs = q_focus, na.rm = TRUE)),
    x_max = max(mean_pred, na.rm = TRUE),
    y_max = max(obs_rate,  na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    m_q = pmax(ifelse(is.finite(x_q), x_q, x_max),
               ifelse(is.finite(y_q), y_q, y_max),
               na.rm = TRUE),
    
    m_max = m_q + pmax(0.15 * m_q, 0.05),
    
    m_max = pmin(pmax(m_max, min_cap), max_cap)
  )

# 7) Panel-specific breaks/labels
make_breaks <- function(maxv) {
  # Choose step based on maxv
  step <- if (maxv <= 0.5) 0.1 else if (maxv <= 1) 0.2 else if (maxv <= 2) 0.5 else if (maxv <= 10) 2 else if (maxv <= 25) 5 else 10
  seq(0, maxv, by = step)
}

make_labels <- function(maxv) {
  # show one decimal for very small ranges
  if (maxv <= 1) label_number(accuracy = 0.1) else label_number(accuracy = 1)
}

x_scales <- lapply(panel_levels, function(lbl) {
  maxv <- panel_limits$m_max[match(lbl, panel_limits$outcome_label)]
  scale_x_continuous(
    limits = c(0, maxv),
    breaks = make_breaks(maxv),
    labels = make_labels(maxv),
    expand = expansion(mult = c(0.02, 0.04))
  )
})

y_scales <- lapply(panel_levels, function(lbl) {
  maxv <- panel_limits$m_max[match(lbl, panel_limits$outcome_label)]
  scale_y_continuous(
    limits = c(0, maxv),
    breaks = make_breaks(maxv),
    labels = make_labels(maxv),
    expand = expansion(mult = c(0.02, 0.04))
  )
})

# 8) Theme
theme_calib <- theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14, color = "#1F2328", hjust = 0.5),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10, color = "#1F2328"),
    
    axis.title.x = element_text(
      size = 11,
      face = "bold",        
      color = "#1F2328",
      margin = margin(t = 6)
    ),
    axis.title.y = element_text(
      size = 11,
      face = "bold",       
      color = "#1F2328",
      margin = margin(r = 6)
    ),
    axis.text = element_text(size = 9, color = "#1F2328"),
    axis.ticks = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line  = element_line(linewidth = 0.35, color = "#2B2B2B"),
    
    panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
    panel.grid.major.x = element_line(color = "#F5F7FA", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 9.5, color = "#1F2328"),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    
    plot.margin = margin(10, 12, 8, 10)
  )

# 9) Plot
p_calib <- ggplot(calib_df, aes(x = mean_pred, y = obs_rate, color = model, group = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.35, color = "#8B949E") +
  geom_line(linewidth = 0.85, alpha = 0.95) +
  scale_color_manual(values = model_cols) +
  labs(
    title = "Calibration Curve",
    x = "Predicted risk (%)",
    y = "Observed event rate (%)"
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_calib +
  ggh4x::facet_wrap2(~ outcome_label, nrow = 4, ncol = 5, scales = "free", axes = "all") +
  ggh4x::facetted_pos_scales(x = x_scales, y = y_scales)

p_calib

ggsave("FigS3_lightgbm_calibration.pdf", p_calib, width = 10, height = 7)




###### Figure S4 LightGBM模型DCA结果
###### DCA
suppressPackageStartupMessages({
  library(tidyverse)
  library(glue)
  library(scales)
})

# 0) Paths
base_dir   <- "LightGBM"
dca_dir    <- file.path(base_dir, "dca_outputs", "cv")

# 1) Outcomes (12) + labels
outcome_label_map_12 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease"
)
outcomes_12  <- names(outcome_label_map_12)
panel_levels <- unname(outcome_label_map_12)

# 2) Models (4) + labels
model_types_4 <- c(
  "model1_base",
  "model2_clinical",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)
model_levels_4 <- unname(model_type_labels[model_types_4])

# 3) Colors
model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3", # not used
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)

fill_color <- scales::alpha(model_cols[["Model 5: Clin+Exam (longitudinal)"]], 0.25)

# 4) Read DCA CSVs
read_dca_one <- function(outcome) {
  f <- file.path(dca_dir, glue("dca_{outcome}.csv"))
  if (!file.exists(f)) {
    warning("Missing DCA file: ", f)
    return(NULL)
  }
  
  readr::read_csv(f, show_col_types = FALSE) %>%
    filter(model_type %in% c(model_types_4, "treat_all", "treat_none")) %>%
    transmute(
      outcome       = outcome,
      model_type    = as.character(model_type),
      thresholds    = as.numeric(thresholds) * 100,  # -> %
      NB            = as.numeric(NB),
      sNB           = as.numeric(sNB) * 100,          # -> %
      prevalence    = as.numeric(prevalence),
      outcome_label = factor(outcome_label_map_12[outcome], levels = panel_levels)
    ) %>%
    filter(!is.na(thresholds), !is.na(sNB))
}

dca_df_raw <- bind_rows(lapply(outcomes_12, read_dca_one))

# 5) Keep model and reference data, add labels
dca_models_raw <- dca_df_raw %>%
  filter(model_type %in% model_types_4) %>%
  mutate(
    model = model_type_labels[model_type],
    model = factor(model, levels = model_levels_4)
  )

dca_ref <- dca_df_raw %>%
  filter(model_type %in% c("treat_all", "treat_none")) %>%
  group_by(outcome_label, model_type) %>%
  arrange(thresholds) %>%
  ungroup()

# 6) Global threshold truncation
model5_type <- "model5_clinical_dynamic_exam"

thr_limits <- dca_models_raw %>%
  filter(model_type == model5_type) %>%
  group_by(outcome_label) %>%
  arrange(thresholds) %>%
  summarise(
    thr_cut = suppressWarnings(min(thresholds[sNB <= 0], na.rm = TRUE)),
    thr_max_data = suppressWarnings(max(thresholds, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(
    thr_cut = ifelse(is.finite(thr_cut), thr_cut, thr_max_data),
    thr_max = pmin(thr_cut + 2, thr_max_data),
    thr_max = pmax(thr_max, 5)
  )

dca_models_raw <- dca_models_raw %>%
  left_join(thr_limits, by = "outcome_label") %>%
  filter(thresholds <= thr_max) %>%
  select(-thr_cut, -thr_max_data, -thr_max)

dca_ref <- dca_ref %>%
  left_join(thr_limits, by = "outcome_label") %>%
  filter(thresholds <= thr_max) %>%
  select(-thr_cut, -thr_max_data, -thr_max)

# 7) Enforce "floor at 0" and "absorbing at 0" for model curves
# Rule:
#   - any sNB < 0 -> 0
#   - once sNB reaches 0 (after flooring), keep sNB=0 for all later thresholds
enforce_floor_absorb0 <- function(df) {
  df <- df %>% arrange(thresholds)
  s <- pmax(df$sNB, 0)
  first_zero <- which(s <= 0)[1]
  if (!is.na(first_zero)) {
    s[first_zero:length(s)] <- 0
  }
  df$sNB <- s
  df
}

dca_models <- dca_models_raw %>%
  group_by(outcome_label, model_type) %>%
  group_modify(~ enforce_floor_absorb0(.x)) %>%
  ungroup() %>%
  mutate(
    # keep ordering stable
    model = factor(model_type_labels[model_type], levels = model_levels_4)
  )

# 8) Ribbon between Model 4 and Model 5 (use adjusted sNB)
dca_ribbon <- dca_models %>%
  filter(model %in% c("Model 4: Clin+Exam (baseline)",
                      "Model 5: Clin+Exam (longitudinal)")) %>%
  select(outcome_label, thresholds, model, sNB) %>%
  pivot_wider(names_from = model, values_from = sNB) %>%
  drop_na(`Model 4: Clin+Exam (baseline)`, `Model 5: Clin+Exam (longitudinal)`) %>%
  arrange(outcome_label, thresholds) %>%
  mutate(
    ymin = pmin(`Model 4: Clin+Exam (baseline)`, `Model 5: Clin+Exam (longitudinal)`),
    ymax = pmax(`Model 4: Clin+Exam (baseline)`, `Model 5: Clin+Exam (longitudinal)`)
  )

# 9) Theme
theme_dca <- theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14, color = "#1F2328", hjust = 0.5),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10, color = "#1F2328"),
    
    axis.title.x = element_text(size = 11, face = "bold", color = "#1F2328", margin = margin(t = 6)),
    axis.title.y = element_text(size = 11, face = "bold", color = "#1F2328", margin = margin(r = 6)),
    axis.text = element_text(size = 9, color = "#1F2328"),
    
    axis.ticks = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line  = element_line(linewidth = 0.45, color = "#2B2B2B"),
    
    panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
    panel.grid.major.x = element_line(color = "#F5F7FA", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 9.5, color = "#1F2328"),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    
    plot.margin = margin(10, 12, 8, 10)
  )

# 10) Plot
treat_all_df  <- dca_ref %>% filter(model_type == "treat_all")
treat_none_df <- dca_ref %>% filter(model_type == "treat_none")

p_dca <- ggplot() +
  # Treat-none
  geom_line(
    data = treat_none_df,
    aes(x = thresholds, y = sNB, group = outcome_label),
    color = "#6B7280",
    linewidth = 0.45,
    linetype = "solid",
    alpha = 0.95
  ) +
  # Treat-all
  geom_line(
    data = treat_all_df,
    aes(x = thresholds, y = sNB, group = outcome_label),
    color = "#9CA3AF",
    linewidth = 0.45,
    linetype = "solid",
    alpha = 0.95
  ) +
  # Ribbon between Model 4 and Model 5
  geom_ribbon(
    data = dca_ribbon,
    aes(x = thresholds, ymin = ymin, ymax = ymax, group = outcome_label),
    fill = fill_color
  ) +
  # Model curves (ALL SOLID)
  geom_line(
    data = dca_models,
    aes(x = thresholds, y = sNB, color = model, group = interaction(outcome_label, model)),
    linewidth = 0.7,
    alpha = 0.95,
    linetype = "solid"
  ) +
  facet_wrap(~ outcome_label, nrow = 3, ncol = 4, scales = "free_x") +
  scale_color_manual(values = model_cols[model_levels_4]) +
  coord_cartesian(ylim = c(-1, NA)) +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.04))) +
  labs(
    x = "Threshold probability (%)",
    y = "Standardized net benefit (%)"
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_dca

p_dca

ggsave("FigS4_lightgbm_DCA.pdf", p_dca, width = 10, height = 7)






###### Figure S5 Logistic模型校准曲线
###### 校准曲线
# 0) Paths
base_dir   <- "Logistic"
pred_dir   <- file.path(base_dir, "outer_cv_predictions")

# 1) Outcomes 
outcome_label_map <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  ischemic_stroke          = "Ischaemic stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  arterial_disease         = "Arterial disease",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease",
  parkinson                = "Parkinson's disease",
  dementia                 = "All-cause dementia",
  cancer_all               = "All-cause cancer",
  liver_cancer             = "Liver cancer",
  lung_cancer              = "Lung cancer",
  kidney_cancer            = "Kidney cancer"
)
outcomes_20  <- names(outcome_label_map)
panel_levels <- unname(outcome_label_map)

# 2) Models (5) 
model_types_5 <- c(
  "model1_base",
  "model2_clinical",
  "model3_dynamic",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)

# 3) Colors
model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3",
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)

# 4) Read prediction file
read_pred_one <- function(outcome, model_type) {
  f <- file.path(pred_dir, sprintf("%s_%s_outer5fold_predictions.csv", outcome, model_type))
  if (!file.exists(f)) {
    warning("Missing prediction file: ", f)
    return(NULL)
  }
  readr::read_csv(f, show_col_types = FALSE) %>%
    transmute(
      outcome = outcome,
      model_type = model_type,
      actual = as.numeric(actual),
      pred   = as.numeric(pred_raw)
    ) %>%
    filter(!is.na(actual), !is.na(pred)) %>%
    mutate(pred = pmin(pmax(pred, 0), 1))
}

pred_df <- bind_rows(lapply(outcomes_20, function(oc) {
  bind_rows(lapply(model_types_5, function(mt) read_pred_one(oc, mt)))
}))

# 5) Quantile-binned calibration lines
get_calib_bins <- function(df_one, n_bins = 20) {
  if (nrow(df_one) < n_bins) return(NULL)
  
  df_one %>%
    mutate(bin = ntile(pred, n_bins)) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(pred) * 100,
      obs_rate  = mean(actual) * 100,
      .groups = "drop"
    ) %>%
    arrange(mean_pred)
}

n_bins <- 20

calib_df <- pred_df %>%
  group_by(outcome, model_type) %>%
  group_modify(~{
    out <- get_calib_bins(.x, n_bins = n_bins)
    if (is.null(out)) return(tibble())
    out
  }) %>%
  ungroup() %>%
  mutate(
    outcome_label = factor(outcome_label_map[outcome], levels = panel_levels),
    model = factor(model_type_labels[model_type],
                   levels = unname(model_type_labels[model_types_5]))
  )

# 6) Better per-panel axis maxima
q_focus <- 0.99          
min_cap <- 0.25          
max_cap <- 100

panel_limits <- calib_df %>%
  group_by(outcome_label) %>%
  summarise(
    x_q = suppressWarnings(quantile(mean_pred, probs = q_focus, na.rm = TRUE)),
    y_q = suppressWarnings(quantile(obs_rate,  probs = q_focus, na.rm = TRUE)),
    x_max = max(mean_pred, na.rm = TRUE),
    y_max = max(obs_rate,  na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    m_q = pmax(ifelse(is.finite(x_q), x_q, x_max),
               ifelse(is.finite(y_q), y_q, y_max),
               na.rm = TRUE),
    
    m_max = m_q + pmax(0.15 * m_q, 0.05),
    
    m_max = pmin(pmax(m_max, min_cap), max_cap)
  )

# 7) Panel-specific breaks/labels
make_breaks <- function(maxv) {
  # Choose step based on maxv
  step <- if (maxv <= 0.5) 0.1 else if (maxv <= 1) 0.2 else if (maxv <= 2) 0.5 else if (maxv <= 10) 2 else if (maxv <= 25) 5 else 10
  seq(0, maxv, by = step)
}

make_labels <- function(maxv) {
  # show one decimal for very small ranges
  if (maxv <= 1) label_number(accuracy = 0.1) else label_number(accuracy = 1)
}

x_scales <- lapply(panel_levels, function(lbl) {
  maxv <- panel_limits$m_max[match(lbl, panel_limits$outcome_label)]
  scale_x_continuous(
    limits = c(0, maxv),
    breaks = make_breaks(maxv),
    labels = make_labels(maxv),
    expand = expansion(mult = c(0.02, 0.04))
  )
})

y_scales <- lapply(panel_levels, function(lbl) {
  maxv <- panel_limits$m_max[match(lbl, panel_limits$outcome_label)]
  scale_y_continuous(
    limits = c(0, maxv),
    breaks = make_breaks(maxv),
    labels = make_labels(maxv),
    expand = expansion(mult = c(0.02, 0.04))
  )
})

# 8) Theme
theme_calib <- theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14, color = "#1F2328", hjust = 0.5),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10, color = "#1F2328"),
    
    axis.title.x = element_text(
      size = 11,
      face = "bold",        
      color = "#1F2328",
      margin = margin(t = 6)
    ),
    axis.title.y = element_text(
      size = 11,
      face = "bold",       
      color = "#1F2328",
      margin = margin(r = 6)
    ),
    axis.text = element_text(size = 9, color = "#1F2328"),
    axis.ticks = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line  = element_line(linewidth = 0.35, color = "#2B2B2B"),
    
    panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
    panel.grid.major.x = element_line(color = "#F5F7FA", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 9.5, color = "#1F2328"),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    
    plot.margin = margin(10, 12, 8, 10)
  )

# 9) Plot
p_calib <- ggplot(calib_df, aes(x = mean_pred, y = obs_rate, color = model, group = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.35, color = "#8B949E") +
  geom_line(linewidth = 0.85, alpha = 0.95) +
  scale_color_manual(values = model_cols) +
  labs(
    title = "Calibration Curve",
    x = "Predicted risk (%)",
    y = "Observed event rate (%)"
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_calib +
  ggh4x::facet_wrap2(~ outcome_label, nrow = 4, ncol = 5, scales = "free", axes = "all") +
  ggh4x::facetted_pos_scales(x = x_scales, y = y_scales)

p_calib

ggsave("FigS5_logistic_calibration.pdf", p_calib, width = 10, height = 7)



###### Figure S6 Logistic模型DCA结果
###### DCA
suppressPackageStartupMessages({
  library(tidyverse)
  library(glue)
  library(scales)
})

# 0) Paths
base_dir   <- "Logistic"
dca_dir    <- file.path(base_dir, "dca_outputs", "cv")

# 1) Outcomes (12) + labels
outcome_label_map_12 <- c(
  mi                       = "Myocardial infarction",
  afib_flutter             = "Atrial fibrillation",
  cor_pulmonale            = "Cor pulmonale",
  chf                      = "Heart failure",
  stroke                   = "All-cause stroke",
  hemorrhagic_stroke       = "Haemorrhagic stroke",
  copd                     = "COPD",
  liver_fibrosis_cirrhosis = "Liver cirrhosis",
  liver_failure            = "Liver failure",
  renal_failure            = "Renal failure",
  diabetes                 = "Diabetes",
  thyroid_disease          = "Thyroid disease"
)
outcomes_12  <- names(outcome_label_map_12)
panel_levels <- unname(outcome_label_map_12)

# 2) Models (4) + labels
model_types_4 <- c(
  "model1_base",
  "model2_clinical",
  "model4_clinical_baseline_exam",
  "model5_clinical_dynamic_exam"
)

model_type_labels <- c(
  model1_base                   = "Model 1: Base",
  model2_clinical               = "Model 2: Clinical",
  model3_dynamic                = "Model 3: Only longitudinal",
  model4_clinical_baseline_exam = "Model 4: Clin+Exam (baseline)",
  model5_clinical_dynamic_exam  = "Model 5: Clin+Exam (longitudinal)"
)
model_levels_4 <- unname(model_type_labels[model_types_4])

# 3) Colors
model_cols <- c(
  "Model 1: Base"                      = "#969696",
  "Model 2: Clinical"                  = "#abd9e9",
  "Model 3: Only longitudinal"         = "#4393c3", # not used
  "Model 4: Clin+Exam (baseline)"      = "#b2abd2",
  "Model 5: Clin+Exam (longitudinal)"  = "#fc9272"
)

fill_color <- scales::alpha(model_cols[["Model 5: Clin+Exam (longitudinal)"]], 0.25)

# 4) Read DCA CSVs
read_dca_one <- function(outcome) {
  f <- file.path(dca_dir, glue("dca_{outcome}.csv"))
  if (!file.exists(f)) {
    warning("Missing DCA file: ", f)
    return(NULL)
  }
  
  readr::read_csv(f, show_col_types = FALSE) %>%
    filter(model_type %in% c(model_types_4, "treat_all", "treat_none")) %>%
    transmute(
      outcome       = outcome,
      model_type    = as.character(model_type),
      thresholds    = as.numeric(thresholds) * 100,  # -> %
      NB            = as.numeric(NB),
      sNB           = as.numeric(sNB) * 100,          # -> %
      prevalence    = as.numeric(prevalence),
      outcome_label = factor(outcome_label_map_12[outcome], levels = panel_levels)
    ) %>%
    filter(!is.na(thresholds), !is.na(sNB))
}

dca_df_raw <- bind_rows(lapply(outcomes_12, read_dca_one))

# 5) Keep model and reference data, add labels
dca_models_raw <- dca_df_raw %>%
  filter(model_type %in% model_types_4) %>%
  mutate(
    model = model_type_labels[model_type],
    model = factor(model, levels = model_levels_4)
  )

dca_ref <- dca_df_raw %>%
  filter(model_type %in% c("treat_all", "treat_none")) %>%
  group_by(outcome_label, model_type) %>%
  arrange(thresholds) %>%
  ungroup()

# 6) Global threshold truncation
model5_type <- "model5_clinical_dynamic_exam"

thr_limits <- dca_models_raw %>%
  filter(model_type == model5_type) %>%
  group_by(outcome_label) %>%
  arrange(thresholds) %>%
  summarise(
    thr_cut = suppressWarnings(min(thresholds[sNB <= 0], na.rm = TRUE)),
    thr_max_data = suppressWarnings(max(thresholds, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(
    thr_cut = ifelse(is.finite(thr_cut), thr_cut, thr_max_data),
    thr_max = pmin(thr_cut + 2, thr_max_data),
    thr_max = pmax(thr_max, 5)
  )

dca_models_raw <- dca_models_raw %>%
  left_join(thr_limits, by = "outcome_label") %>%
  filter(thresholds <= thr_max) %>%
  select(-thr_cut, -thr_max_data, -thr_max)

dca_ref <- dca_ref %>%
  left_join(thr_limits, by = "outcome_label") %>%
  filter(thresholds <= thr_max) %>%
  select(-thr_cut, -thr_max_data, -thr_max)

# 7) Enforce "floor at 0" and "absorbing at 0" for model curves
# Rule:
#   - any sNB < 0 -> 0
#   - once sNB reaches 0 (after flooring), keep sNB=0 for all later thresholds
enforce_floor_absorb0 <- function(df) {
  df <- df %>% arrange(thresholds)
  s <- pmax(df$sNB, 0)
  first_zero <- which(s <= 0)[1]
  if (!is.na(first_zero)) {
    s[first_zero:length(s)] <- 0
  }
  df$sNB <- s
  df
}

dca_models <- dca_models_raw %>%
  group_by(outcome_label, model_type) %>%
  group_modify(~ enforce_floor_absorb0(.x)) %>%
  ungroup() %>%
  mutate(
    # keep ordering stable
    model = factor(model_type_labels[model_type], levels = model_levels_4)
  )

# 8) Ribbon between Model 4 and Model 5 (use adjusted sNB)
dca_ribbon <- dca_models %>%
  filter(model %in% c("Model 4: Clin+Exam (baseline)",
                      "Model 5: Clin+Exam (longitudinal)")) %>%
  select(outcome_label, thresholds, model, sNB) %>%
  pivot_wider(names_from = model, values_from = sNB) %>%
  drop_na(`Model 4: Clin+Exam (baseline)`, `Model 5: Clin+Exam (longitudinal)`) %>%
  arrange(outcome_label, thresholds) %>%
  mutate(
    ymin = pmin(`Model 4: Clin+Exam (baseline)`, `Model 5: Clin+Exam (longitudinal)`),
    ymax = pmax(`Model 4: Clin+Exam (baseline)`, `Model 5: Clin+Exam (longitudinal)`)
  )

# 9) Theme
theme_dca <- theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14, color = "#1F2328", hjust = 0.5),
    
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10, color = "#1F2328"),
    
    axis.title.x = element_text(size = 11, face = "bold", color = "#1F2328", margin = margin(t = 6)),
    axis.title.y = element_text(size = 11, face = "bold", color = "#1F2328", margin = margin(r = 6)),
    axis.text = element_text(size = 9, color = "#1F2328"),
    
    axis.ticks = element_line(linewidth = 0.35, color = "#2B2B2B"),
    axis.line  = element_line(linewidth = 0.45, color = "#2B2B2B"),
    
    panel.grid.major.y = element_line(color = "#F0F3F6", linewidth = 0.30),
    panel.grid.major.x = element_line(color = "#F5F7FA", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 9.5, color = "#1F2328"),
    legend.key.width = unit(18, "pt"),
    legend.spacing.x = unit(10, "pt"),
    
    plot.margin = margin(10, 12, 8, 10)
  )

# 10) Plot
treat_all_df  <- dca_ref %>% filter(model_type == "treat_all")
treat_none_df <- dca_ref %>% filter(model_type == "treat_none")

p_dca <- ggplot() +
  # Treat-none
  geom_line(
    data = treat_none_df,
    aes(x = thresholds, y = sNB, group = outcome_label),
    color = "#6B7280",
    linewidth = 0.45,
    linetype = "solid",
    alpha = 0.95
  ) +
  # Treat-all
  geom_line(
    data = treat_all_df,
    aes(x = thresholds, y = sNB, group = outcome_label),
    color = "#9CA3AF",
    linewidth = 0.45,
    linetype = "solid",
    alpha = 0.95
  ) +
  # Ribbon between Model 4 and Model 5
  geom_ribbon(
    data = dca_ribbon,
    aes(x = thresholds, ymin = ymin, ymax = ymax, group = outcome_label),
    fill = fill_color
  ) +
  # Model curves (ALL SOLID)
  geom_line(
    data = dca_models,
    aes(x = thresholds, y = sNB, color = model, group = interaction(outcome_label, model)),
    linewidth = 0.7,
    alpha = 0.95,
    linetype = "solid"
  ) +
  facet_wrap(~ outcome_label, nrow = 3, ncol = 4, scales = "free_x") +
  scale_color_manual(values = model_cols[model_levels_4]) +
  coord_cartesian(ylim = c(-1, NA)) +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.04))) +
  labs(
    x = "Threshold probability (%)",
    y = "Standardized net benefit (%)"
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_dca

p_dca

ggsave("FigS6_logistic_DCA.pdf", p_dca, width = 10, height = 7)







