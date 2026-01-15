#!/usr/bin/env Rscript
# =============================================================================
# DeepGS_Independent_Test.CNN.v5.1.R
# =============================================================================

suppressPackageStartupMessages(library(DeepGS))
suppressPackageStartupMessages(library(mxnet))

# 记录程序总开始时间
start_total_time <- Sys.time()

# ============================= 参数解析 =============================
args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  cat("Usage: Rscript DeepGS_Independent_Test.CNN.v5.1.R --tr-snp=FILE --tr-phe=FILE --te-snp=FILE --te-phe=FILE ...\n")
  q("no")
}

parse_arg <- function(arg_name, default = NULL, required = FALSE) {
  pattern <- paste0("^--", arg_name, "=")
  value <- gsub(pattern, "", grep(pattern, args, value = TRUE))
  if (length(value) == 0) {
    if (required) stop(paste("Missing required argument:", arg_name))
    return(default)
  }
  return(value)
}

tr_snp <- parse_arg("tr-snp", required = TRUE)
tr_phe <- parse_arg("tr-phe", required = TRUE)
te_snp <- parse_arg("te-snp", required = TRUE)
te_phe <- parse_arg("te-phe", required = TRUE)

num_round <- as.integer(parse_arg("num_round", default = "6000"))
lr <- as.numeric(parse_arg("lr", default = "0.01"))
array_batch_size <- as.integer(parse_arg("batch-size", default = "30"))
randomseeds <- as.integer(parse_arg("randomseeds", default = "42"))

# ============================= 1. 数据读取与预处理 =============================
cat(sprintf("[%s] Loading Data...\n", Sys.time()))

tr_snp_raw <- read.table(tr_snp, header = TRUE, row.names = 1, sep = "\t", check.names = FALSE)
te_snp_raw <- read.table(te_snp, header = TRUE, row.names = 1, sep = "\t", check.names = FALSE)

common_snps <- intersect(rownames(tr_snp_raw), rownames(te_snp_raw))
Markers_train <- t(as.matrix(tr_snp_raw[common_snps, ]))
Markers_test  <- t(as.matrix(te_snp_raw[common_snps, ]))

tr_phe_df <- read.csv(tr_phe, row.names = 1, stringsAsFactors = FALSE)
te_phe_df <- read.csv(te_phe, row.names = 1, stringsAsFactors = FALSE)

common_train <- intersect(rownames(tr_phe_df), rownames(Markers_train))
Markers_train <- Markers_train[common_train, ]
tr_phe_df <- tr_phe_df[common_train, , drop = FALSE]

common_test <- intersect(rownames(te_phe_df), rownames(Markers_test))
Markers_test <- Markers_test[common_test, ]
te_phe_df <- te_phe_df[common_test, , drop = FALSE]

# --- 1.1 提取并保存测试集 Accession ID 以便后续输出对齐 ---
# 保存当前最终参与测试的样本行名（Accession ID）
test_accession_names <- rownames(te_phe_df)

cat(sprintf("Training Set: %d Individuals, %d Markers\n", nrow(Markers_train), ncol(Markers_train)))
cat(sprintf("Testing Set : %d Individuals, %d Markers\n", nrow(Markers_test), ncol(Markers_test)))
# ============================= 2. CNN 设置 =============================
# 将全连接层激活函数从 sigmoid 改为 tanh 以匹配 Normalization
cnnFrame <- list(
  conv_kernel = c("1*18"), conv_num_filter = c(8), conv_stride = c("1*1"),
  pool_act_type = c("relu"), pool_type = c("max"), pool_kernel = c("1*4"),
  pool_stride = c("1*4"), fullayer_num_hidden = c(32, 1),
  fullayer_act_type = c("tanh"),  # 适配标准化数据
  drop_float = c(0.2, 0.1, 0.05)
)

markerImage <- paste0("1*", ncol(Markers_train))

# ============================= 3. 多性状循环预测 =============================
all_predicted_values <- list()
all_observed_values <- list()
results <- data.frame(Trait = character(), Pearson_r = numeric(), Time_Seconds = numeric(), stringsAsFactors = FALSE)

traits <- colnames(tr_phe_df)

for(trait in traits) {
  cat(sprintf("\n>>> Processing Trait: %s\n", trait))
  start_trait_time <- Sys.time()
  
  y_train_raw <- tr_phe_df[[trait]]
  y_test_obs  <- te_phe_df[[trait]]
  
  # --- 表型 Normalization 处理 ---
  y_mean <- mean(y_train_raw, na.rm = TRUE)
  y_sd   <- sd(y_train_raw, na.rm = TRUE)
  
  # 1. Z-score 标准化
  y_train_normalized <- ((y_train_raw - y_mean) / y_sd) / 3
  y_train_scaled <- pmin(pmax(y_train_normalized, -0.99), 0.99)
  
  # --- 3.1 划分训练集和验证集 (10% 验证集) ---
  n_total_train <- length(y_train_scaled)
  set.seed(randomseeds)
  valid_idx <- sample(1:n_total_train, floor(n_total_train * 0.1))
  
  # --- 3.2 训练 DeepGS CNN 模型 ---
  model_deep <- train_deepGSModel(
    trainMat = Markers_train[-valid_idx, ], 
    trainPheno = y_train_scaled[-valid_idx],
    validMat = Markers_train[valid_idx, ], 
    validPheno = y_train_scaled[valid_idx],
    markerImage = markerImage, 
    cnnFrame = cnnFrame,
    device_type = "cpu", 
    gpuNum = "0", 
    eval_metric = "mae",
    num_round = num_round, 
    array_batch_size = array_batch_size, 
    learning_rate = lr,
    momentum = 0.5, 
    wd = 0.00001, 
    randomseeds = randomseeds, 
    initializer_idx = 0.01,
    verbose = FALSE
  )
  
  # --- 3.3 预测与反向还原 (Inverse Normalization) ---
  pred_scaled <- as.vector(predict_GSModel(model_deep, Markers_test, markerImage))
  
  # 还原
  pred_final <- (pred_scaled * 3) * y_sd + y_mean
  
  all_predicted_values[[trait]] <- pred_final
  all_observed_values[[trait]] <- y_test_obs
  
  # --- 3.4 评估 (Pearson Correlation) ---
  if (sd(pred_final, na.rm = TRUE) == 0) {
      cat("  --> Warning: Zero variance in predictions. Setting r to NA.\n")
      pearson_r <- NA
  } else {
      pearson_r <- cor(y_test_obs, pred_final, use = "complete.obs", method = "pearson")
  }
  
  end_trait_time <- Sys.time()
  elapsed_trait_time <- as.numeric(difftime(end_trait_time, start_trait_time, units = "secs"))
  
  cat(sprintf("  --> Result: Pearson r = %.4f | Time: %.2f seconds\n", 
              pearson_r, elapsed_trait_time))
  
  results <- rbind(results, data.frame(
    Trait = trait,
    Pearson_r = pearson_r,
    Time_Seconds = elapsed_trait_time 
  ))
}

# ============================= 4. 输出结果与总计时 =============================
output_file <- "DeepGS_CNN_CPU_Results.csv"
write.csv(results, output_file, row.names = FALSE)

# --- 4.1 输出预测与观察值调试文件 (保持 v4.0 逻辑) ---
cat("\n--- 开始输出调试文件 ---\n")
df_predicted <- as.data.frame(all_predicted_values)
df_observed <- as.data.frame(all_observed_values)

# --- 4.1.1 (v5.1 新增): 将 Accession ID 注入输出文件的首列 ---
df_predicted_with_id <- cbind(Accession = test_accession_names, df_predicted)
df_observed_with_id <- cbind(Accession = test_accession_names, df_observed)

predicted_output_file <- gsub("\\.csv$", "_predicted.tsv", output_file)
observed_output_file <- gsub("\\.csv$", "_observed.tsv", output_file)

# 修改：写入包含 ID 的新数据框，保持原有的文件名和输出逻辑
write.table(df_predicted_with_id, file = predicted_output_file, sep = "\t", row.names = FALSE, quote = FALSE)
write.table(df_observed_with_id, file = observed_output_file, sep = "\t", row.names = FALSE, quote = FALSE)
cat("--- 调试文件输出完成 (已合并 Accession 列) ---\n")

end_total_time <- Sys.time()
total_elapsed_time <- difftime(end_total_time, start_total_time, units = "mins")
cat(sprintf("\nDone! Total Elapsed Time: %.2f minutes\n", as.numeric(total_elapsed_time)))
