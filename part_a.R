library(DoubleML)
library(mlr3)
library(mlr3learners)
library(data.table)
library(ggplot2)

set.seed(321)

samples <- 2000
confounders <- 1000
true_theta_0 <- 0.5

z_mat <- matrix(rnorm(samples * confounders), nrow = samples)

# sparse confounding effects (only uses some confounders)
g0_sparse <- (sin(z_mat[,1]) * z_mat[,2]) + (z_mat[,3] * z_mat[,4])^2 + rowSums(z_mat[, 5:10])
m0_sparse <- (cos(z_mat[,5]) * z_mat[,6]) + (z_mat[,7] - z_mat[,8])^2 + rowSums(z_mat[, 9:15] * 0.5)

# dense confounding effects (uses all confounders)
weights <- 1 / (1:confounders)
g0_dense <- 20 * tanh((z_mat %*% weights)) + (z_mat[,1] * z_mat[,2])
m0_dense <- 8 * sin((z_mat %*% rev(weights))) + (z_mat[,3] ^ 2)

run_sim <- function(g0, m0, label) {
  d <- m0 + rnorm(samples, sd = 0.5)
  y <- d * true_theta_0 + g0 + rnorm(samples, sd = 0.5)
  
  dt <- as.data.table(cbind(y = y, d = d, z_mat))
  colnames(dt) <- c("y", "d", paste0("z", 1:confounders))
  obj_data <- DoubleMLData$new(dt, y_col = "y", d_cols = "d")
  
  learners <- list(
    lasso  = lrn("regr.cv_glmnet", alpha = 1),       # Lasso (L1)
    elastic_net   = lrn("regr.cv_glmnet", alpha = 0.5),     # Elastic Net (L1 + L2)
    rf     = lrn("regr.ranger", num.trees = 500),    # Random Forest
    xgb    = lrn("regr.xgboost", nrounds = 100, eta = 0.1) # XGBoost
  )
  
  results <- lapply(names(learners), function(name) {
    cat("Fitting", name, "on", label, "...\n")
    dml <- DoubleMLPLR$new(obj_data, ml_l = learners[[name]], ml_m = learners[[name]])
    dml$fit()
    data.table(Scenario = label, Learner = name, Estimate = dml$coef, SE = dml$se)
  })
  
  return(rbindlist(results))
}

final_output_double_ml <- rbind(
  run_sim(g0_sparse, m0_sparse, "Sparse Nonlinear"),
  run_sim(g0_dense, m0_dense, "Dense Nonlinear")
)

get_ols <- function(g0, m0, label) {
  d <- m0 + rnorm(samples, sd = 0.5)
  y <- d * true_theta_0 + g0 + rnorm(samples, sd = 0.5)
  
  dt <- as.data.table(cbind(y = y, d = d, z_mat))
  
  ols_model <- lm(y ~ ., data = dt)
  
  est <- coef(ols_model)["d"]
  se  <- summary(ols_model)$coefficients["d", "Std. Error"]
  
  return(data.table(Scenario = label, Learner = "OLS (Baseline)", 
                    Estimate = est, SE = se))
}

ols_sparse <- get_ols(g0_sparse, m0_sparse, "Sparse Nonlinear")
ols_dense <- get_ols(g0_dense, m0_dense, "Dense Nonlinear")

final_output <- rbind(final_output_double_ml, ols_sparse)

final_output[, Bias := Estimate - 0.5]
final_output[, Abs_Bias := abs(Estimate - 0.5)]

ggplot(final_output, aes(x = Learner, y = Estimate, color = Learner)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = Estimate - 1.96*SE, ymax = Estimate + 1.96*SE), width = 0.2) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red") +
  facet_wrap(~Scenario) +
  theme_bw(base_size = 12) +
  labs(title = "DoubleML Learner Comparison",
       y = "Coefficient Estimate", x = "") +
  theme(legend.position = "")

ggsave("part_a.png", 
       width = 7, 
       height = 5, 
       units = "in", 
       dpi = 300)

sparse_results <- final_output[Scenario == "Sparse Nonlinear"]
dense_results  <- final_output[Scenario == "Dense Nonlinear"]

print(sparse_results)
print(dense_results)
