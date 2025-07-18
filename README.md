# attrition_randomforest
Predicting employee attrition using a Random Forest model in R, with SMOTE resampling, threshold tuning, SHAP explanations, and performance visualizations. 

---

## Project Summary

This project builds a classification model to predict employee attrition using the `attrition` dataset from the `modeldata` package. The workflow includes:

- **Data cleaning & preprocessing**
- **SMOTE** to address class imbalance
- **Random Forest** with parameter tuning to mitigate overfitting
- **Model evaluation** via ROC, PR curves, and threshold tuning
- **SHAP explanations** for model interpretability
- **Interactive Shiny app** to explore threshold-performance tradeoffs
- **Contour risk plot** visualizing risk zones by tenure and salary

---

## Key Findings

- SMOTE increased the model's sensitivity but required careful tuning to avoid overfitting.
- Lowering the classification threshold (e.g., 0.15) improved recall for identifying "quitting" employees.
- Salary and tenure were the strongest predictors of attrition.
- The model’s test AUC was approximately 0.69, indicating modest generalizability.

---

## Files

- `employee_attrition_model_LeahGlassow.Rmd`: Reproducible analysis in R Markdown
- `attrition_randomforest_isobar.R`: End-to-end R script
- Visual output files 
- `LICENSE`: MIT License
- `README.md`: This file

---

##  To Run the Project

In RStudio:

```r
# Install required packages if not already installed
install.packages(c("dplyr", "randomForest", "smotefamily", "pROC", "caret", "DALEX", "ingredients", "ggplot2", "shiny", "patchwork"))

# Load the R Markdown
rmarkdown::render("attrition_rf.Rmd")

# Launch the Shiny app (optional)
shiny::runApp("shiny_app.R")
