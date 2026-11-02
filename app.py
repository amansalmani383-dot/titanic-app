import streamlit as st
import joblib
import pandas as pd
FEATURES = ["pclass", "sex", "age", "fare"]
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')
st.title("Titanic Survival Prediction")
pclass = st.selectbox("Ticket Class", [1,2,3])
sex_option = st.radio("Sex", ["female","male"])
age = st.slider("Age", 1, 80, 30)
fare = st.slider("Fare", 0, 500, 30)
sex = 1 if sex_option=="female" else 0
if st.button("Predict"):
    import pandas as pd
    df = pd.DataFrame([[pclass, sex, age, fare]], columns=FEATURES)
    model = joblib.load('model.joblib')
    prob = model.predict_proba(df)[0][1]*100
    st.write(f"Survival: {prob:.1f}%")