# Income Classification Based on Demographic and Work Attributes

This project predicts whether a person's income is above 50K using demographic and work-related attributes from the Adult Income dataset.

## Features
- Dataset preview and missing value analysis
- Exploratory Data Analysis (EDA)
- Preprocessing and feature engineering
- Model comparison: Logistic Regression, KNN, SVM
- Confusion matrix and classification report
- Single record prediction form
- Batch prediction demo with downloadable sample CSV

## Files
- `app.py` - Streamlit dashboard
- `train_model.py` - training script to generate saved model artifacts
- `utils.py` - shared data loading, preprocessing, plotting, and modeling functions
- `adult.data` - dataset
- `artifacts/` - saved model and model results after training

## How to run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Project workflow
1. Load Adult Income dataset
2. Clean missing values and strip extra spaces
3. Create engineered features
4. Split into train/test sets
5. Train multiple classification models
6. Compare metrics
7. Use best model for live prediction

## Suggested viva explanation
This project predicts whether a person earns more than 50K annually based on demographic and employment information. The dashboard shows the full machine learning pipeline from preprocessing to evaluation and final prediction.
