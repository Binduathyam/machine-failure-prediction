
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="centered"
)

# Load model
model = pickle.load(
    open("../model/model.pkl", "rb")
)

# Load dataset for scaling
df = pd.read_csv("../predictive_maintenance.csv")

# Preprocessing
df = df.drop(
    columns=["UDI", "Product ID"]
)

df["Type"] = df["Type"].replace({
    "L": 0,
    "M": 1,
    "H": 2
})

# Features
X = df.drop(
    columns=["Target", "Failure Type"]
)

# Scaling
scaler = StandardScaler()
scaler.fit(X)

# Sidebar
st.sidebar.title("⚙️ About Project")

st.sidebar.info(
    """
    This project predicts machine failure
    using industrial sensor data.

    Features used:
    - Air Temperature
    - Process Temperature
    - Rotational Speed
    - Torque
    - Tool Wear
    - Machine Type
    """
)

# Title
st.title("⚙️ Predictive Maintenance Dashboard")

st.markdown(
    "### Predict Machine Failure Using Sensor Data"
)

st.divider()

# Input Section
st.subheader("📥 Enter Machine Details")

# Machine Type
machine_type = st.selectbox(
    "Machine Type",
    ["L (Low)", "M (Medium)", "H (High)"]
)

# Convert type
type_map = {
    "L (Low)": 0,
    "M (Medium)": 1,
    "H (High)": 2
}

machine_type = type_map[machine_type]

# Inputs
air_temp = st.number_input(
    "Air Temperature [K]",
    value=300.0
)

process_temp = st.number_input(
    "Process Temperature [K]",
    value=310.0
)

speed = st.number_input(
    "Rotational Speed [rpm]",
    value=1500
)

torque = st.number_input(
    "Torque [Nm]",
    value=40.0
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    value=100
)

st.divider()

# Prediction
if st.button("🔍 Predict Machine Health"):

    features = np.array([[
        machine_type,
        air_temp,
        process_temp,
        speed,
        torque,
        tool_wear
    ]])

    # Apply scaling
    scaled_features = scaler.transform(
        features
    )

    # Prediction
    prediction = model.predict(
        scaled_features
    )

    # Health Score
    health_score = max(
        0,
        100 - (tool_wear * 0.2)
    )

    st.subheader(
        "📊 Machine Health Score"
    )

    st.progress(
        int(health_score)
    )

    st.write(
        f"Health Score: {health_score:.2f}%"
    )

    # Result
    if prediction[0] == 1:

        st.error(
            "⚠️ Machine Failure Risk"
        )

        st.warning(
            "Recommendation: Perform maintenance immediately."
        )

    else:

        st.success(
            "✅ Machine Healthy"
        )

        st.info(
            "Recommendation: Machine is operating normally."
        )

