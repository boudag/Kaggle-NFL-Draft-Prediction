# 🏈 Kaggle NFL Draft Prediction

![NFL Draft](https://img.shields.io/badge/Kaggle-Competition-blue.svg) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensembling-brightgreen.svg)

This repository contains an end-to-end Machine Learning pipeline designed to predict whether college football players will be drafted into the NFL based on their physical attributes and NFL Combine performance. 

## 📊 The Data
The dataset contains historical NFL Combine records including:
- **Physical Metrics:** Height, Weight, Age
- **Athletic Drills:** 40-Yard Dash, Vertical Jump, Broad Jump, 3-Cone Drill, Bench Press
- **Categorical Data:** College/School, Position

## 🧠 Machine Learning Architecture

This project was built using two distinct architectures, separated by branches:

### 1. `master` Branch (The Monolithic Pipeline)
A highly optimized, linear pipeline that utilizes a standard Ensembling Strategy.
- **Base Models:** XGBoost, LightGBM, CatBoost
- **Hyperparameter Tuning:** Optuna (100 Trials per model)
- **Meta-Learner:** Logistic Regression
- **Performance:** Reached **0.840 ROC-AUC** on the online leaderboard!

### 2. `v8_distributed_pipeline` Branch (The Cloud Architecture)
A heavily distributed architecture designed to bypass Google Colab runtime limits by training models in parallel across multiple cloud tabs.
- **Added Diversity:** Random Forest, K-Nearest Neighbors, TabNet (Deep Learning)
- **The Ultimate Ensembler:** A fully dynamic stacking script (`05_Ultimate_Ensembler.ipynb`) that automatically parses all generated OOF predictions and executes a 4-way evaluation between:
  1. Logistic Regression
  2. Greedy Hill-Climbing
  3. Rank Averaging
  4. AutoGluon Non-Linear Stacking

## 📁 Repository Structure
- `notebooks/`: Contains all Jupyter Notebooks (both the Monolithic and Distributed pipelines depending on your branch).
- `src/`: Core Python modules for data loading, preprocessing, feature engineering, and model training.
- `data/`: Place your `train.csv` and `test.csv` here.
- `docs/`: Technical notes and future experimental ideas (Pseudo-Labeling, Target Encoding, Data Leakage tracking).

## 🚀 How to Run (Distributed Branch)
1. Upload the `notebooks/01` through `07` to Google Colab in separate tabs.
2. Upload `train.csv` and `test.csv` to each tab.
3. Run all tabs in parallel to generate the Out-Of-Fold (OOF) CSV predictions.
4. Upload all resulting CSV predictions to the `05_Ultimate_Ensembler.ipynb` tab and execute it to generate the final `submission.csv`!
