# Load libraries
library(dplyr)
library(readr)
library(haven)
library(modeldata)
library(randomForest)
library(janitor)
library(smotefamily)
library(caret)
library(pROC)
library(forecast)
library(PRROC)
library(ggplot2)
library(viridis)
library(DALEX)
library(ingredients)
library(patchwork)
library(shiny)

# Data preparation and cleaning
attdf <- as.data.frame(attrition)
set.seed(123)
years <- 1989:2010
attdf$Year <- sample(years, size = nrow(attdf), replace = TRUE)
attdf <- janitor::clean_names(attdf)

# Ensure target is a factor
attdf$attrition <- factor(attdf$attrition, levels = c("No", "Yes"))

# Train-test split
set.seed(42)
train_idx <- sample(1:nrow(attdf), 0.7 * nrow(attdf))
train <- attdf[train_idx, ]
test <- attdf[-train_idx, ]

# Ensure all factor levels are consistent across train and test
factor_vars <- names(attdf)[sapply(attdf, function(x) is.factor(x) || is.character(x))]
for (var in factor_vars) {
  levels_combined <- union(levels(factor(attdf[[var]])), levels(factor(attdf[[var]])))
  train[[var]] <- factor(train[[var]], levels = levels_combined)
  test[[var]] <- factor(test[[var]], levels = levels_combined)
}

# fit random forest model
rf_model <- randomForest(
  attrition ~ monthly_income + years_at_company + stock_option_level,
  data = train,
  ntree = 500,
  importance = TRUE
)

pred_initial <- predict(rf_model, newdata = test)
confusionMatrix(pred_initial, test$attrition, positive = "Yes")

#smote
# Encode factor vars to numeric with consistent mapping
train_copy <- train
for (col in names(train_copy)) {
  if (is.factor(train_copy[[col]]) || is.character(train_copy[[col]])) {
    train_copy[[col]] <- as.numeric(factor(train_copy[[col]]))
  }
}

# Select predictors
X <- train_copy[, c("monthly_income", "years_at_company",
                     "stock_option_level")]
y <- as.numeric(train$attrition) - 1  # 0 = No, 1 = Yes

# Apply SMOTE
smote_result <- SMOTE(X, y, K = 5, dup_size = 1)
smote_data <- smote_result$data
smote_data$attrition <- factor(smote_data$class, labels = c("No", "Yes"))
smote_data$class <- NULL

# Fit model on SMOTE data
rf_smote <- randomForest(
  attrition ~ monthly_income + years_at_company +
   stock_option_level,
  data = smote_data,
  ntree = 200,
  nodesize=10,
  maxnodes=15,
  importance = TRUE
)

# Prepare test data to match numeric-encoded SMOTE format
test_copy <- test
for (col in names(test_copy)) {
  if (is.factor(test_copy[[col]]) || is.character(test_copy[[col]])) {
    test_copy[[col]] <- as.numeric(factor(test_copy[[col]],
                                          levels = levels(factor(train[[col]]))))
  }
}

# Predict on numeric-encoded test set
pred_smote <- predict(rf_smote, newdata = test_copy)
confusionMatrix(pred_smote, test$attrition, positive = "Yes")
windows()
varImpPlot(rf_smote)
ggsave("varimpsmote.png", plot = last_plot())

# ROC-AUC

# Get predicted probabilities for the positive class
probs <- predict(rf_smote, newdata = test_copy, type = "prob")[, "Yes"]

# Generate ROC curve
roc_obj <- roc(test$attrition, probs, levels = c("No", "Yes"), direction = "<")

# Plot it
windows()
plot(roc_obj, main = "ROC Curve", col = "blue", lwd = 2)
ggsave("roc_curve.png", plot = last_plot())
auc_value <- auc(roc_obj)
text(0.6, 0.4, paste("AUC =", round(auc_value, 3)), col = "blue")


# Predict probabilities for the positive class 
scores <- predict(rf_smote, newdata = test_copy, type = "prob")[, "Yes"]
labels <- ifelse(test$attrition == "Yes", 1, 0)

# Generate PR curve
pr <- pr.curve(scores.class0 = scores[labels == 1],
               scores.class1 = scores[labels == 0],
               curve = TRUE)

# Plot
windows()
plot(pr, main = "Precision-Recall Curve", col = "darkorange", lwd = 2)
ggsave("prcurve.png", plot = last_plot())

#catch more quitters and compare thresholds 

thresholds <- seq(0.1, 0.4, by = 0.05)
true <- test$attrition
probs <- predict(rf_smote, newdata = test_copy, type = "prob")[, "Yes"]

library(caret)

for (t in thresholds) {
  pred <- factor(ifelse(probs > t, "Yes", "No"), levels = c("No", "Yes"))
  cm <- confusionMatrix(pred, true, positive = "Yes")
  cat("\nThreshold:", t,
      "\nSensitivity:", round(cm$byClass["Sensitivity"], 3),
      "| Precision:", round(cm$byClass["Precision"], 3),
      "| F1:", round(cm$byClass["F1"], 3), "\n")
}


#summarize: predictors of employee attrition

# Predict probabilities
probs <- predict(rf_smote, newdata = test_copy, type = "prob")[, "Yes"]
true <- test$attrition

# Loop through thresholds
thresholds <- seq(0.1, 0.4, by = 0.05)
results <- data.frame(
  Threshold = thresholds,
  Sensitivity = NA,
  Precision = NA,
  F1 = NA
)

for (i in seq_along(thresholds)) {
  pred <- factor(ifelse(probs > thresholds[i], "Yes", "No"), levels = c("No", "Yes"))
  cm <- caret::confusionMatrix(pred, true, positive = "Yes")
  results$Sensitivity[i] <- cm$byClass["Sensitivity"]
  results$Precision[i] <- cm$byClass["Precision"]
  results$F1[i] <- cm$byClass["F1"]
}

# sort by F1
results <- results[order(-results$F1), ]
print(results)

# Plot
windows()
plot(results$Threshold, results$Sensitivity, type = "o", col = "red", ylim = c(0, 1),
     ylab = "Score", xlab = "Threshold", main = "Threshold Tuning")
lines(results$Threshold, results$Precision, type = "o", col = "blue")
lines(results$Threshold, results$F1, type = "o", col = "purple")
abline(v = 0.15, lty = 2, col = "darkgray")
legend("bottomleft", legend = c("Sensitivity", "Precision", "F1"),
       col = c("red", "blue", "purple"), lty = 1, bty = "n")
ggsave("thresholds.png", plot = last_plot())

#assess overfit
# Predict on training set (SMOTE data)
train_preds <- predict(rf_smote, newdata = smote_data, type = "prob")[, "Yes"]

# Predict on untouched test set
test_preds <- predict(rf_smote, newdata = test_copy, type = "prob")[, "Yes"]

# Compare AUC
roc_train <- roc(smote_data$attrition, train_preds)
roc_test <- roc(test$attrition, test_preds)

auc(roc_train)  
auc(roc_test)   


#shap

# Select only those columns 
predictor_vars <- c("monthly_income", "years_at_company", "stock_option_level")

# Drop unused levels 
smote_data <- droplevels(smote_data)

# Recreate explainer using numeric SMOTE data
explainer_rf <- explain(
  rf_smote,
  data = smote_data[, predictor_vars],
  y = as.numeric(smote_data$attrition == "Yes"),
  label = "RF"
)

#Prepare test observation 
test_shap <- test[1, predictor_vars, drop = FALSE]

# Convert test observation to numeric in the same way as during SMOTE preprocessing
for (col in predictor_vars) {
  test_shap[[col]] <- as.numeric(factor(test_shap[[col]], levels = levels(factor(train[[col]]))))
}

# Predict SHAP
shap_values <- predict_parts(
  explainer_rf,
  new_observation = test_shap,
  type = "shap"
)

# Plot
windows()
plot(shap_values)
ggsave("shap.png", plot = last_plot())

#isobar contour risk plot

# Subset SMOTE data to include only relevant predictors
smote_data_reduced <- smote_data[, c("monthly_income", "years_at_company", "attrition")]
smote_data_reduced$attrition <- factor(smote_data_reduced$attrition, levels = c("No", "Yes"))

# Train Random Forest model
set.seed(123)
rf_2vars <- randomForest(
  attrition ~ monthly_income + years_at_company,
  data = smote_data_reduced,
  ntree = 500,
  importance = TRUE
)

# Create grid for prediction
x_seq <- seq(min(smote_data_reduced$monthly_income), max(smote_data_reduced$monthly_income), length.out = 100)
y_seq <- seq(min(smote_data_reduced$years_at_company), max(smote_data_reduced$years_at_company), length.out = 60)
grid <- expand.grid(
  monthly_income = x_seq,
  years_at_company = y_seq
)

# Predict attrition risk
grid$predicted_risk <- predict(rf_2vars, newdata = grid, type = "prob")[, "Yes"]

# Plot 

windows()  
ggplot(grid, aes(x = monthly_income, y = years_at_company)) +
  geom_tile(aes(fill = predicted_risk)) +
  stat_contour(aes(z = predicted_risk), color = "white", bins = 10, alpha = 0.7) +
  labs(
    title = "Employee Attrition Risk Contour Map",
    x = "Monthly Income",
    y = "Years at Company"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "right",
    panel.grid = element_blank()
  )+
  geom_point(data = subset(grid, predicted_risk > 0.85), 
             aes(x = monthly_income, y = years_at_company), 
             color = "red", size = 1.5, alpha = 0.7)+
  scale_fill_gradientn(
    colours = c("black", "purple", "orange", "red"),
    name = "Predicted Risk",
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25),
    labels = scales::percent_format(accuracy = 1)
  )
ggsave("isobar.png", plot = last_plot())


#shiny dashboard 

# predicted probabilities 
probs <- predict(rf_smote, newdata = test_copy, type = "prob")[, "Yes"]
true <- test$attrition

ui <- fluidPage(
  titlePanel("Attrition Model Threshold Explorer"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("thresh", "Prediction Threshold", min = 0.1, max = 0.9, value = 0.3, step = 0.01),
      verbatimTextOutput("f1")
    ),
    mainPanel(
      plotOutput("rocPlot"),
      tableOutput("confMatrix")
    )
  )
)

server <- function(input, output) {
  
  output$rocPlot <- renderPlot({
    plot(roc(true, probs), main = "ROC Curve", col = "blue", lwd = 2)
    auc_val <- auc(roc(true, probs))
    text(0.6, 0.4, paste("AUC =", round(auc_val, 3)), col = "blue")
  })
  
  output$confMatrix <- renderTable({
    pred <- factor(ifelse(probs > input$thresh, "Yes", "No"), levels = c("No", "Yes"))
    cm <- caret::confusionMatrix(pred, true, positive = "Yes")
    cm$table
  })
  
  output$f1 <- renderText({
    pred <- factor(ifelse(probs > input$thresh, "Yes", "No"), levels = c("No", "Yes"))
    cm <- caret::confusionMatrix(pred, true, positive = "Yes")
    paste("F1 Score:", round(cm$byClass["F1"], 3))
  })
  
}

# Launch the app
shinyApp(ui = ui, server = server)
