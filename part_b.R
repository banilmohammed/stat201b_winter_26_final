library(grf)
library(ggplot2)

set.seed(321)
samples <- 3000
confounders <- 10
x <- matrix(rnorm(samples * confounders), samples, confounders)

w <- rnorm(samples, sd = 1) 
baseline <- 2 * x[, 1] + 3 * x[, 2]

true_theta_curve <- 4 / (1 + exp(-2 * x[, 3])) 
y <- baseline + (true_theta_curve * (w^2)) + rnorm(samples, sd = 0.2)

y_hat <- predict(regression_forest(x, y))$predictions
w_sq <- w^2
w_sq_hat <- predict(regression_forest(x, w_sq))$predictions

y_res <- y - y_hat
w_sq_res <- w_sq - w_sq_hat

final_forest <- causal_forest(x, y, w_sq, Y.hat = y_hat, W.hat = w_sq_hat)
est_theta <- predict(final_forest)$predictions

w_sq <- w^2
w_sq_hat <- predict(regression_forest(x, w_sq))$predictions
w_sq_res <- w_sq - w_sq_hat

linear_dml_model <- lm(y_res ~ w_sq_res - 1)
avg_linear_effect <- coef(linear_dml_model)

comparison_df <- data.frame(
  X3 = x[, 3],
  True_Theta = true_theta_curve,
  Proposed_Extension = est_theta,
  Standard_Linear_DML = as.numeric(avg_linear_effect)
)

ggplot(comparison_df, aes(x = X3)) +
  geom_line(aes(y = True_Theta, color = "True Curvature (f0)"), size = 1.5) +
  geom_point(aes(y = Proposed_Extension, color = "Non-linear DML Extension"), alpha = 0.4) +
  geom_hline(aes(yintercept = Standard_Linear_DML, color = "Standard Linear DML"), 
             linetype = "dashed", size = 1.2) +
  scale_color_manual(values = c(
    "True Curvature (f0)" = "black", 
    "Non-linear DML Extension" = "#f8766d", 
    "Standard Linear DML" = "blue"
  )) +
  labs(title = "DML Extension Performance",
       y = "Treatment Effect", x = "Confounder Affecting Treatment Effect (X3)",
       color = "Model") +
  theme_minimal()

ggsave("part_b_performance.jpg", 
       width = 7, 
       height = 5, 
       units = "in", 
       dpi = 300)

high_impact <- which.max(x[,3]) # Someone on the right of the S-curve
low_impact  <- which.min(x[,3]) # Someone on the left of the S-curve

d_range <- seq(-2, 2, length.out = 100)
curves <- data.frame(
  Dosage = rep(d_range, 2),
  Impact = c(est_theta[high_impact] * d_range^2, est_theta[low_impact] * d_range^2),
  Group = rep(c("High Modifier (X3 > 0)", "Low Modifier (X3 < 0)"), each = 100)
)

ggplot(curves, aes(x = Dosage, y = Impact, color = Group)) +
  geom_line(size = 1.5) +
  labs(title = "Estimated Individual Dose-Response Curves",
       x = "Dosage (W)", y = "Estimated Effect on Y") +
  theme_minimal()

ggsave("part_b_drc.jpg", 
       width = 7, 
       height = 5, 
       units = "in", 
       dpi = 300)

var_imp <- variable_importance(final_forest)

importance_df <- data.frame(
  Variable = paste0("X", 1:10),
  Importance = as.numeric(var_imp[1:10])
)

ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue", width = 0.7) +
  coord_flip() +  # Makes it easier to read variable names
  theme_minimal() +
  labs(
    title = "Feature Importance on Treatment Effect for Non-linear DML Extension",
    x = "Confounders",
    y = "Importance Score"
  )

ggsave("part_b_feature_imp.jpg", 
       width = 7, 
       height = 5, 
       units = "in", 
       dpi = 300)
