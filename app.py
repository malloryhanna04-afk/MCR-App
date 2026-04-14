import streamlit as st
import pandas as pd
import numpy as np
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
# Train the regression model
# -----------------------------
@st.cache_resource
def train_model(df):
    X = df[["Angle", "Ds", "Dv", "Ratio"]]
    y = df["MCR"]
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# -----------------------------
# UI Inputs
# -----------------------------
angle = st.number_input("Angle (degrees)", min_value=0, max_value=180, value=120)
Dv = st.number_input("Vessel Diameter (mm)", min_value=1.0, max_value=10.0, value=3.0)
Ds = st.number_input("Stent Diameter (mm)", min_value=1.0, max_value=10.0, value=4.0)

ratio = Ds / Dv
st.write(f"Computed Ds/Dv ratio: {ratio:.2f}")

# -----------------------------
# Predict MCR
# -----------------------------
if st.button("Calculate MCR"):
    X_new = np.array([[angle, Ds, Dv, ratio]])
    mcr_pred = model.predict(X_new)[0]
    st.success(f"Predicted MCR: {mcr_pred:.2f}%")

import plotly.express as px

st.subheader("3D Visualization: Angle vs Ds/Dv vs MCR")

fig = px.scatter_3d(
    df,
    x="Angle",
    y="Ratio",
    z="MCR",
    color="Dv",   # you can change this to "Region" if you prefer
    title="3D Plot of Angle, Ds/Dv Ratio, and MCR",
    height=650
)

st.plotly_chart(fig)

import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.subheader("3D Quadratic Surface: Angle vs Ds/Dv vs MCR")

# Extract variables
X = df[["Angle", "Ratio"]]
y = df["MCR"]

# Create quadratic features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_quad = poly.fit_transform(X)

# Fit quadratic regression
quad_model = LinearRegression()
quad_model.fit(X_quad, y)

# Create grid for surface
angle_vals = np.linspace(df["Angle"].min(), df["Angle"].max(), 40)
ratio_vals = np.linspace(df["Ratio"].min(), df["Ratio"].max(), 40)
A, R = np.meshgrid(angle_vals, ratio_vals)

# Prepare grid for prediction
grid = np.column_stack([A.ravel(), R.ravel()])
grid_quad = poly.transform(grid)

# Predict MCR on grid
Z = quad_model.predict(grid_quad).reshape(A.shape)

# Plot surface
fig = go.Figure(data=[go.Surface(
    x=A,
    y=R,
    z=Z,
    colorscale="Viridis"
)])

fig.update_layout(
    title="Quadratic Fit Surface",
    scene=dict(
        xaxis_title="Angle (degrees)",
        yaxis_title="Ds/Dv Ratio",
        zaxis_title="MCR"
    ),
    height=700
)

st.plotly_chart(fig)
