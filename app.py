import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier

st.set_page_config(page_title="Shoppers Intention Prediction", layout="wide")

dtc_model = joblib.load('decision_tree_model.pkl')
cat_model = CatBoostClassifier()
cat_model.load_model('catboost_model.cbm')
encoder = joblib.load('encoder.pkl')

st.title("Online Shoppers Intention Prediction 🛒")

model_choice = st.selectbox(
    "Select Model",
    ("CatBoost", "Decision Tree")
)

col1, col2, col3 = st.columns(3)

with col1:
    Administrative = st.slider("Administrative", 0, 27, 0)
    Administrative_Duration = st.slider("Administrative Duration", 0.0, 3400.0, 0.0)
    Informational = st.slider("Informational", 0, 24, 0)
    Informational_Duration = st.slider("Informational Duration", 0.0, 2550.0, 0.0)
    ProductRelated = st.slider("ProductRelated", 0, 705, 0)
    ProductRelated_Duration = st.slider("ProductRelated Duration", 0.0, 64000.0, 0.0)

with col2:
    BounceRates = st.slider("BounceRates", 0.0, 0.2, 0.0, format="%.4f")
    ExitRates = st.slider("ExitRates", 0.0, 0.2, 0.0, format="%.4f")
    PageValues = st.slider("PageValues", 0.0, 365.0, 0.0, format="%.2f")
    SpecialDay = st.slider("SpecialDay", 0.0, 1.0, 0.0, step=0.2)
    OperatingSystems = st.selectbox("OperatingSystems", [1, 2, 3, 4, 5, 6, 7, 8])
    Browser = st.selectbox("Browser", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

with col3:
    Region = st.selectbox("Region", [1, 2, 3, 4, 5, 6, 7, 8, 9])
    TrafficType = st.selectbox("TrafficType", list(range(1, 21)))
    Month = st.selectbox("Month", ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    VisitorType = st.selectbox("VisitorType", ['Returning_Visitor', 'New_Visitor', 'Other'])
    Weekend = st.checkbox("Weekend")

if st.button("Predict"):
    input_data = pd.DataFrame({
        'Administrative': [Administrative],
        'Administrative_Duration': [Administrative_Duration],
        'Informational': [Informational],
        'Informational_Duration': [Informational_Duration],
        'ProductRelated': [ProductRelated],
        'ProductRelated_Duration': [ProductRelated_Duration],
        'BounceRates': [BounceRates],
        'ExitRates': [ExitRates],
        'PageValues': [PageValues],
        'SpecialDay': [SpecialDay],
        'Month': [Month],
        'OperatingSystems': [OperatingSystems],
        'Browser': [Browser],
        'Region': [Region],
        'TrafficType': [TrafficType],
        'VisitorType': [VisitorType],
        'Weekend': [Weekend]
    })

    if model_choice == "CatBoost":
        prediction = cat_model.predict(input_data)
    else:
        input_data[['Month', 'VisitorType']] = encoder.transform(input_data[['Month', 'VisitorType']])
        prediction = dtc_model.predict(input_data)

    if prediction[0] == 1 or prediction[0] == 'True':
        st.success("The customer will purchase 💸")
    else:
        st.error("Tha customer will not purchase 🚶‍♂️")