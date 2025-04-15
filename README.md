# Telco-Customer-Churn

# 📊 Telco Customer Churn Prediction

This project uses machine learning to predict customer churn for a telecom company. It leverages data preprocessing, feature engineering, SMOTE for handling imbalance, model selection with hyperparameter tuning, and evaluation metrics to deliver robust churn predictions.

---

## 📁 Dataset

- Source: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Rows: ~7,000+
- Features: Customer demographics, services signed up for, account information, churn status

---

## 🔧 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib & Seaborn
- SMOTE (imbalanced-learn)
- Jupyter Notebook

---

## 🔍 Project Workflow

1. **Load Data**
   - Read and clean the CSV
   - Handle missing values

2. **Preprocessing**
   - Encode categorical features
   - Scale numerical features
   - Handle class imbalance with SMOTE

3. **Model Training**
   - Train-test split with stratified sampling
   - Hyperparameter tuning using GridSearchCV
   - Stratified K-Fold Cross Validation
   - Train a `RandomForestClassifier`

4. **Model Evaluation**
   - Classification report
   - Confusion matrix heatmap
   - ROC-AUC score

5. **Model Saving**
   - Final model saved as `churn_model.pkl` using Pickle

---

## 📈 Results

- **Model Used**: Random Forest (GridSearch Optimized)
- **Evaluation Metrics**:
  - Precision, Recall, F1-Score
  - ROC-AUC

---

## 🚀 How to Run

1. Clone the repo or download the `.ipynb` file.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
