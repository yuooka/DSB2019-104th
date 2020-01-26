# Kaggle Data Science Bowl 2019 - 104th solution
This repository contains scripts for 104th place in kaggle competition, "[Data Science Bowl 2019](https://www.kaggle.com/c/data-science-bowl-2019)."
# Overview
## 1. Feature engineering
- User action logs are converted into aggregation features using "pd.groupby"
## 2. Feature selection
- Created features are dropped one by one based on "Null importance"
## 3. Adversarial validation
- Features distribution are compared using "Adversarial validation"
## 4. Build model
- LightGBM is the first choice
- Custom weighted metric for this competition is developed
