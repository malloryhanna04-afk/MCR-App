import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.title("MCR Calculator")

# -----------------------------
# Load your dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("data.xlsx")
    df = df.dropna(subset=["Angle (degrees)"])
    df = df.rename(columns={
        "Angle (degrees)": "Angle",
        "Ds (mm) (stent diameter)": "Ds",
        "Dv (mm) (vessel diameter)": "Dv",
        "Ds/Dv": "Ratio",
        "MCR": "MCR"
    })
    return df

df = load_data()

# -----------------------------
# Train Random Forest model
# -----------------------------
X_rf = df[["Angle", "Ds", "Dv", "Ratio"]]
y_rf = df["MCR"]

rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
rf_model.fit(X_rf, y_rf)

# -----------------------------
# Train Quadratic Regression model
# -----------------------------
X_quad = df[["Angle", "Ratio"]]
y_quad = df["MCR"]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_quad_transformed = poly.fit_transform(X_quad)

quad_model = LinearRegression()
quad_model.fit(X_quad_transformed, y_quad)

# -----------------------------
# UI Inputs
# -----------------------------
angle = st.number_input("Angle (degrees)", min_value=0, max_value=180, value=120)
Dv = st.number_input("Vessel Diameter (mm)", min_value=1.0, max_value=10.0, value=3.0)
Ds = st.number_input("Stent Diameter (mm)", min_value=1.0, max_value=10.0, value=4.0)

ratio = Ds / Dv
st.write(f"Computed Ds/Dv ratio: {ratio:.2f}")

# -----------------------------
# Model selection
# -----------------------------
model_choice = st.radio(
    "Choose prediction model:",
    ["Quadratic Fit", "Random Forest"]
)

# -----------------------------
# Predict MCR using selected model
# -----------------------------
if st.button("Calculate MCR"):
    if model_choice == "Quadratic Fit":
        quad_input = poly.transform([[angle, ratio]])
        mcr_pred = quad_model.predict(quad_input)[0]

    elif model_choice == "Random Forest":
        rf_input = np.array([[angle, Ds, Dv, ratio]])
        mcr_pred = rf_model.predict(rf_input)[0]

    st.session_state.mcr_pred = mcr_pred
    st.session_state.angle = angle
    st.session_state.ratio = ratio
    st.session_state.model_choice = model_choice

    st.success(f"Predicted MCR ({model_choice}): {mcr_pred:.2f}%")


# -----------------------------
# 3D Surface using Random Forest model
# -----------------------------
st.subheader("3D Model Surface (Random Forest)")

angle_vals = np.linspace(df["Angle"].min(), df["Angle"].max(), 40)
ratio_vals = np.linspace(df["Ratio"].min(), df["Ratio"].max(), 40)
A, R = np.meshgrid(angle_vals, ratio_vals)

grid_full = np.column_stack([
    A.ravel(),
    np.full(A.size, Ds),
    np.full(A.size, Dv),
    R.ravel()
])

Z_rf = rf_model.predict(grid_full).reshape(A.shape)

fig = go.Figure(data=[go.Surface(
    x=A,
    y=R,
    z=Z_rf,
    colorscale="Viridis",
    colorbar=dict(title="MCR (%)")
)])

# Add dot only if Random Forest was used
if "mcr_pred" in st.session_state and st.session_state.model_choice == "Random Forest":
    fig.add_trace(go.Scatter3d(
        x=[st.session_state.angle],
        y=[st.session_state.ratio],
        z=[st.session_state.mcr_pred],
        mode="markers",
        marker=dict(size=8, color="red"),
        name="Your MCR"
    ))


fig.update_layout(
    title="Random Forest Model Surface",
    scene=dict(
        xaxis_title="Angle (degrees)",
        yaxis_title="Ds/Dv Ratio",
        zaxis_title="MCR"
    ),
    height=700
)

st.plotly_chart(fig)

# -----------------------------
# 3D Quadratic Surface (Prediction Model)
# -----------------------------
st.subheader("3D Quadratic Surface: Angle vs Ds/Dv vs MCR")

grid_quad = poly.transform(np.column_stack([A.ravel(), R.ravel()]))
Z_quad = quad_model.predict(grid_quad).reshape(A.shape)

fig = go.Figure(data=[go.Surface(
    x=A,
    y=R,
    z=Z_quad,
    colorscale="Viridis",
    colorbar=dict(title="MCR (%)")
)])

# Add dot only if Quadratic Fit was used
if "mcr_pred" in st.session_state and st.session_state.model_choice == "Quadratic Fit":
    fig.add_trace(go.Scatter3d(
        x=[st.session_state.angle],
        y=[st.session_state.ratio],
        z=[st.session_state.mcr_pred],
        mode="markers",
        marker=dict(size=8, color="red"),
        name="Your MCR"
    ))


# Add the user's predicted point
if "mcr_pred" in st.session_state:
    fig.add_trace(go.Scatter3d(
        x=[st.session_state.angle],
        y=[st.session_state.ratio],
        z=[st.session_state.mcr_pred],
        mode="markers",
        marker=dict(size=8, color="red"),
        name="Your MCR"
    ))

fig.update_layout(
    title="Quadratic Fit Surface (Prediction Model)",
    scene=dict(
        xaxis_title="Angle (degrees)",
        yaxis_title="Ds/Dv Ratio",
        zaxis_title="MCR"
    ),
    height=700
)

st.plotly_chart(fig)
