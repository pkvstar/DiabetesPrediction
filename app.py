import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

data = pd.read_csv(r"./diabetes_prediction_dataset.csv")
data["gender"] = data["gender"].map({"Male": 1, "Female": 2, "Other": 3})
data["smoking_history"] = data["smoking_history"].map({
    "never": 1, "No Info": 2, "current": 3, "former": 4, "ever": 5, "not current": 6
})

y = data['diabetes']
x = data.drop("diabetes", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)

st.title("Diabetes Risk Assessment")

st.markdown("### Enter Patient Details Below")
with st.form("diabetes_form"):
    
    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox("Age", options=list(range(0, 101)), index=25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        bmi = st.selectbox("BMI (Body Mass Index)", options=[round(x, 1) for x in np.arange(10.0, 50.5, 0.5)], index=30)
        hypertension = st.selectbox("Hypertension", [0, 1])

    with col2:
        smoking_history = st.selectbox("Smoking History", ["Never", "No Info", "Current", "Former", "Ever", "Not Current"])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        hba1c = st.selectbox("HbA1c Level (Glycated Hemoglobin)", options=[round(x, 1) for x in np.arange(4.0, 15.5, 0.1)], index=15)
        blood_glucose = st.selectbox("Blood Glucose Level", options=list(range(50, 251, 5)), index=10)

    st.markdown("</div>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="Analyze Risk")

def assess_diabetes_risk(model, input_features):
    prediction = model.predict(input_features)
    if prediction[0] == 1:
        return (
            "🚨 **High Risk of Diabetes Detected!**\n\n"
            "It's important to consult a healthcare provider as soon as possible. "
            "Here are a few recommendations:\n"
            "- Schedule a check-up to confirm the diagnosis.\n"
            "- Maintain a balanced diet and monitor your blood sugar levels.\n"
            "- Incorporate regular physical activity into your routine.\n\n"
            "Stay proactive and take control of your health! 💪"
        )
    else:
        return (
            "✅ **Low Risk of Diabetes Detected!**\n\n"
            "Keep up the good work maintaining your health! To stay on track:\n"
            "- Continue healthy eating habits.\n"
            "- Stay physically active.\n"
            "- Monitor your health regularly.\n\n"
            "Prevention is the key to long-term well-being. 🌟"
        )


if submit_button:
    gender_map = {"Male": 1, "Female": 2, "Other": 3}
    smoking_map = {"Never": 1, "No Info": 2, "Current": 3, "Former": 4, "Ever": 5, "Not Current": 6}
    
    input_data = np.array([[age, gender_map[gender], smoking_map[smoking_history], 
                            hypertension, heart_disease, bmi, hba1c, blood_glucose]])
    
    risk_message = assess_diabetes_risk(rf_model, input_data)
    
    st.markdown(
        f"<div style='padding: 20px; border: 2px solid #007bff; border-radius: 10px; "
        f"text-align: left; font-size: 1.2em; color: #212529; background-color: #f8f9fa;'>"
        f"{risk_message}</div>",
        unsafe_allow_html=True
    )


importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = x.columns

plt.figure(figsize=(8, 6))
plt.title("Feature Importance", fontsize=16, color="#1A5276")
plt.bar(range(x.shape[1]), importances[indices], align="center", color="#5DADE2")
plt.xticks(range(x.shape[1]), features[indices], rotation=45, ha='right', fontsize=10)
plt.tight_layout()

