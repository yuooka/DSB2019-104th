# Kaggle Data Science Bowl 2019 - 104th solution by [Y. O.](https://www.kaggle.com/yuooka)
This repository contains scripts for 104th place in kaggle competition, "[Data Science Bowl 2019](https://www.kaggle.com/c/data-science-bowl-2019)."
# Overview
### 1. Feature engineering
  - User action logs are converted into aggregation features using "pd.groupby"
### 2. Feature selection
  - Created features are dropped one by one based on "Null importance"
### 3. Adversarial validation
  - Features distribution are compared using "Adversarial validation"
### 4. Build model
  - LightGBM is the first choice
  - Custom weighted metric for this competition is developed
  
# Overall pipeline
![DSB2019_2](https://user-images.githubusercontent.com/60316115/73186790-aee48100-4163-11ea-92d5-20401585de9a.png)
