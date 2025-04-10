
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="DreamCracker", layout="centered")

st.title("Dream11 Grand League Predictor - DreamCracker")
st.markdown("### Predict match outcomes based on stats and strategy")

# Sample input data
st.sidebar.header("Input Player Stats")

def user_input_features():
    avg_runs = st.sidebar.slider("Average Runs", 0, 100, 40)
    strike_rate = st.sidebar.slider("Strike Rate", 50, 200, 120)
    wickets = st.sidebar.slider("Wickets Taken", 0, 10, 2)
    economy = st.sidebar.slider("Bowling Economy", 3.0, 12.0, 7.5)
    form = st.sidebar.selectbox("Recent Form", ["Good", "Average", "Poor"])
    is_captain = st.sidebar.checkbox("Is Captain?", False)
    
    form_value = {"Good": 2, "Average": 1, "Poor": 0}[form]
    
    data = {
        "avg_runs": avg_runs,
        "strike_rate": strike_rate,
        "wickets": wickets,
        "economy": economy,
        "form": form_value,
        "is_captain": int(is_captain)
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Dummy model logic (Random prediction for now)
model = RandomForestClassifier()
X_dummy = pd.DataFrame({
    "avg_runs": np.random.randint(0, 100, 100),
    "strike_rate": np.random.randint(50, 200, 100),
    "wickets": np.random.randint(0, 10, 100),
    "economy": np.random.uniform(3, 12, 100),
    "form": np.random.randint(0, 3, 100),
    "is_captain": np.random.randint(0, 2, 100),
})
y_dummy = np.random.randint(0, 2, 100)
model.fit(X_dummy, y_dummy)

prediction = model.predict(input_df)
result = "Likely to perform well!" if prediction[0] == 1 else "May underperform, pick wisely!"

st.subheader("Prediction:")
st.success(result)

st.markdown("---")
st.markdown("**Disclaimer:** This is a prototype based on random data. For entertainment purposes only.")
