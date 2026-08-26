# Online Shoppers Purchasing Intention Prediction 🛒

## 👥 Team Members
* Salma Mohamed
* Joussyana Sabry
* Jessika Sabry
  
## 📌 Project Overview
This machine learning project aims to predict whether an online website visitor will complete a transaction and make a purchase (`Revenue = True` or `False`) based on their browsing behavior, session duration, and other Google Analytics metrics.

## 📊 Dataset
We utilized the **Online Shoppers Intention** dataset. It contains both numerical and categorical features representing the types of pages visited, bounce rates, exit rates, page values, and visitor demographics. 
*Note:* The dataset is highly imbalanced, with approximately 84% of sessions resulting in no purchase and only 16% resulting in a purchase.

## ⚙️ Workflow & Methodology
* **Exploratory Data Analysis (EDA):** Visualized feature distributions and extracted key business insights (e.g., the strong correlation between `PageValues` and purchasing intent).
* **Data Preprocessing:** Handled duplicate records, encoded categorical features, and applied selective scaling for specific models.
* **Model Training:** Trained and evaluated multiple models including Logistic Regression, LightGBM, XGBoost, Decision Tree, SVM, Random Forest and CatBoost.
* **Hyperparameter Tuning:** Applied `GridSearchCV` to optimize model parameters (like `max_depth`, `learning_rate`, and `n_estimators`) to prevent overfitting.
* **Evaluation Metric Focus:** Prioritized **Recall** and **F1-Score** to minimize false negatives and ensure we capture as many potential buyers as possible.

## 🏆 Top Performing Models
After evaluation, the two best models selected for deployment are:
* **CatBoost Classifier:** Provided the best overall balance (Highest F1-Score).
* **Decision Tree (Entropy):** Achieved the highest Recall for identifying purchasing customers.

## 💻 Web App (UI)
A user-friendly web application was built using **Streamlit**. It features interactive sliders and dropdown menus for easy input and allows users to choose between the CatBoost and Decision Tree models for real-time predictions.

### How to Run Locally
1. Install the required Python libraries:
`pip install streamlit pandas scikit-learn catboost`

2. Run the Streamlit app:
`streamlit run app.py`
