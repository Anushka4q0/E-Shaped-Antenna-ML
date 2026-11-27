import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.interpolate import RBFInterpolator

st.set_page_config(page_title="E-Shaped Antenna ML Estimator", layout="wide")

REQUIRED = ["Frequency_GHz", "Lpatch_mm", "Wpatch_mm",
            "S11_dB", "VSWR", "Gain_dB", "Directivity_dB"]
TARGETS = ["S11_dB", "VSWR", "Gain_dB", "Directivity_dB"]

# Template: column headers only
TEMPLATE_CSV = (
    "Frequency_GHz,Lpatch_mm,Wpatch_mm,S11_dB,VSWR,Gain_dB,Directivity_dB\n"
)


class ThinPlateSpline:
    """Exact RBF interpolation (thin-plate spline) on scaled inputs."""
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.lo, self.hi = X.min(axis=0), X.max(axis=0)
        self.model = RBFInterpolator(
            self._scale(X), np.asarray(y, dtype=float),
            kernel="thin_plate_spline")
        return self

    def _scale(self, X):
        return (np.asarray(X, dtype=float) - self.lo) / (self.hi - self.lo)

    def predict(self, X):
        return self.model(self._scale(X))


def make_models():
    return {
        "Linear Regression": lambda: make_pipeline(
            StandardScaler(), LinearRegression()),
        "Polynomial Deg 2": lambda: make_pipeline(
            StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1e-6)),
        "Polynomial Deg 3": lambda: make_pipeline(
            StandardScaler(), PolynomialFeatures(3), Ridge(alpha=1e-4)),
        "Gaussian Process": lambda: make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0])
                       + WhiteKernel(1e-3),
                normalize_y=True, n_restarts_optimizer=2, random_state=42)),
        "SVR (RBF)": lambda: TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                SVR(kernel="rbf", C=100, epsilon=0.01, gamma="scale")),
            transformer=StandardScaler()),
        "RBF Interpolation": lambda: ThinPlateSpline(),
    }


def validate(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, missing
    return df[REQUIRED].dropna().reset_index(drop=True), []


@st.cache_data(show_spinner=False)
def train_all(df):
    X = df[["Lpatch_mm", "Wpatch_mm"]]
    L_MIN, L_MAX = X["Lpatch_mm"].min(), X["Lpatch_mm"].max()
    W_MIN, W_MAX = X["Wpatch_mm"].min(), X["Wpatch_mm"].max()
    interior = ((df["Lpatch_mm"] > L_MIN) & (df["Lpatch_mm"] < L_MAX) &
                (df["Wpatch_mm"] > W_MIN) & (df["Wpatch_mm"] < W_MAX)).to_numpy()

    models = make_models()
    loo = LeaveOneOut()
    rows = []

    for target in TARGETS:
        y = df[target]
        y_arr = y.to_numpy()
        for name, factory in models.items():
            preds = np.zeros(len(y))
            for tr, te in loo.split(X):
                m = factory()
                m.fit(X.iloc[tr], y.iloc[tr])
                preds[te] = np.ravel(m.predict(X.iloc[te]))
            rows.append({
                "Target": target,
                "Model": name,
                "R2_all": r2_score(y_arr, preds),
                "RMSE_all": np.sqrt(mean_squared_error(y_arr, preds)),
                "R2_interior": r2_score(y_arr[interior], preds[interior]),
                "MAE_interior": mean_absolute_error(y_arr[interior], preds[interior]),
                "RMSE_interior": np.sqrt(mean_squared_error(
                    y_arr[interior], preds[interior])),
            })

    results_df = pd.DataFrame(rows)

    best_models, best_names = {}, {}
    for target in TARGETS:
        sub = results_df[results_df["Target"] == target]
        best = sub.sort_values("RMSE_interior").iloc[0]["Model"]
        best_names[target] = best
        m = models[best]()
        m.fit(X, df[target])
        best_models[target] = m

    return results_df, best_models, best_names


# ------------------------------------------------------------
# Sidebar: template download and dataset upload
# ------------------------------------------------------------
st.sidebar.header("Dataset")
st.sidebar.download_button(
    "Download CSV template",
    TEMPLATE_CSV,
    file_name="antenna_dataset_template.csv",
    mime="text/csv",
)
st.sidebar.caption("Use the column names from the template. "
                   "Upload a CSV with at least 6 rows, including points "
                   "inside the Lpatch and Wpatch range.")

up = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if up is None:
    st.title("E-Shaped Microstrip Patch Antenna - ML Estimator")
    st.info("Download the template from the sidebar, or upload your dataset "
            "CSV to start. Required columns: " + ", ".join(REQUIRED))
    st.stop()

try:
    raw = pd.read_csv(up)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

df, missing = validate(raw)
if df is None:
    st.error(f"CSV is missing required columns: {missing}")
    st.stop()

if len(df) < 6:
    st.error("The dataset needs at least 6 rows to train the models.")
    st.stop()

L_MIN, L_MAX = float(df["Lpatch_mm"].min()), float(df["Lpatch_mm"].max())
W_MIN, W_MAX = float(df["Wpatch_mm"].min()), float(df["Wpatch_mm"].max())

interior_count = (
    (df["Lpatch_mm"] > L_MIN) & (df["Lpatch_mm"] < L_MAX) &
    (df["Wpatch_mm"] > W_MIN) & (df["Wpatch_mm"] < W_MAX)
).sum()
if interior_count < 1:
    st.error("The dataset needs points inside the Lpatch and Wpatch range "
             "(not only on the edges) to evaluate the models.")
    st.stop()

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
st.title("E-Shaped Microstrip Patch Antenna - ML Estimator (2.4 GHz)")

with st.spinner("Training and evaluating models..."):
    results_df, best_models, best_names = train_all(df)

tab1, tab2, tab3 = st.tabs(["Estimate", "Model comparison", "Dataset"])

# ------------------------------------------------------------
# Tab 1: Estimate
# ------------------------------------------------------------
with tab1:
    st.write(f"Valid range: Lpatch **{L_MIN}-{L_MAX} mm**, "
             f"Wpatch **{W_MIN}-{W_MAX} mm**")

    c1, c2 = st.columns(2)
    L = c1.number_input("Lpatch (mm)", min_value=L_MIN, max_value=L_MAX,
                        value=(L_MIN + L_MAX) / 2, step=0.1)
    W = c2.number_input("Wpatch (mm)", min_value=W_MIN, max_value=W_MAX,
                        value=(W_MIN + W_MAX) / 2, step=0.1)

    new = pd.DataFrame({"Lpatch_mm": [L], "Wpatch_mm": [W]})
    cols = st.columns(len(TARGETS))
    for col, target in zip(cols, TARGETS):
        value = float(np.ravel(best_models[target].predict(new))[0])
        col.metric(target, f"{value:.4f}", help=f"Model: {best_names[target]}")

    st.subheader("Map of estimated values across the range")
    target_sel = st.selectbox("Parameter", TARGETS)
    gl = np.linspace(L_MIN, L_MAX, 60)
    gw = np.linspace(W_MIN, W_MAX, 60)
    GL, GW = np.meshgrid(gl, gw)
    grid = pd.DataFrame({"Lpatch_mm": GL.ravel(), "Wpatch_mm": GW.ravel()})
    Z = np.ravel(best_models[target_sel].predict(grid)).reshape(GL.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    cs = ax.contourf(GL, GW, Z, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax, label=target_sel)
    ax.scatter(df["Lpatch_mm"], df["Wpatch_mm"], c="white", edgecolors="k",
               label="Dataset points")
    ax.scatter([L], [W], c="red", marker="x", s=120, label="Your input")
    ax.set_xlabel("Lpatch (mm)")
    ax.set_ylabel("Wpatch (mm)")
    ax.set_title(f"{target_sel} ({best_names[target_sel]})")
    ax.legend(loc="upper right")
    st.pyplot(fig)

# ------------------------------------------------------------
# Tab 2: Model comparison
# ------------------------------------------------------------
with tab2:
    st.subheader("Interior RMSE (in-between points, lower is better)")
    pivot = results_df.pivot(index="Model", columns="Target",
                             values="RMSE_interior")
    st.dataframe(pivot.round(4))

    st.subheader("Full leave-one-out results")
    st.dataframe(results_df.round(4), use_container_width=True)

    st.subheader("Selected model per parameter")
    st.write(pd.DataFrame.from_dict(best_names, orient="index",
                                    columns=["Best model"]))

    st.download_button("Download evaluation CSV",
                       results_df.to_csv(index=False),
                       file_name="ML_Model_Evaluation.csv",
                       mime="text/csv")

# ------------------------------------------------------------
# Tab 3: Dataset
# ------------------------------------------------------------
with tab3:
    st.subheader("Dataset used for training")
    st.write(f"{len(df)} rows")
    st.dataframe(df, use_container_width=True)
