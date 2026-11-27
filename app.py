import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import StringIO

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

BUILTIN_CSV = """Frequency_GHz,Lpatch_mm,Wpatch_mm,S11_dB,VSWR,Gain_dB,Directivity_dB
2.4,34,46,-6.96343327,2.62692078,4.17,5.62
2.4,35,46,-10.59694995,1.837783388,4.365,5.815
2.4,36,46,-13.43183109,1.541341741,4.45,5.9
2.4,37,46,-10.76175159,1.815616285,4.425,5.875
2.4,38,46,-7.691198992,2.404343919,4.29,5.74
2.4,34,47,-7.244143726,2.535461466,4.26,5.77
2.4,35,47,-11.13145546,1.768568303,4.44,5.95
2.4,36,47,-14.03765195,1.495829435,4.51,6.02
2.4,37,47,-10.87120854,1.801310471,4.47,5.98
2.4,38,47,-7.693007167,2.403846417,4.32,5.83
2.4,34,48,-7.218373128,2.543545004,4.35,5.88
2.4,35,48,-11.65207847,1.708025338,4.515,6.045
2.4,36,48,-14.69656738,1.451430905,4.57,6.1
2.4,37,48,-11.01458895,1.783055187,4.515,6.045
2.4,38,48,-7.919504979,2.343440052,4.35,5.88
2.4,34,49,-7.345271351,2.504319759,4.44,5.95
2.4,35,49,-12.03200468,1.667608502,4.59,6.1
2.4,36,49,-15.51488929,1.402670431,4.63,6.14
2.4,37,49,-11.12111586,1.769836113,4.56,6.07
2.4,38,49,-7.111326753,2.577785509,4.38,5.89
2.4,34,50,-7.938379945,2.338572068,4.53,5.98
2.4,35,50,-12.52362481,1.619492249,4.665,6.115
2.4,36,50,-16.1362543,1.369731474,4.69,6.14
2.4,37,50,-11.45137594,1.730622098,4.605,6.055
2.4,38,50,-7.767982683,2.383434701,4.41,5.86
"""


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
# Sidebar: choose data source
# ------------------------------------------------------------
st.sidebar.header("Dataset")
source = st.sidebar.radio("Data source",
                          ["Built-in dataset", "Upload my own CSV"])

if source == "Upload my own CSV":
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up is None:
        st.info("Upload a CSV file in the sidebar, or switch to the built-in dataset.")
        st.stop()
    raw = pd.read_csv(up)
else:
    raw = pd.read_csv(StringIO(BUILTIN_CSV))

df, missing = validate(raw)
if df is None:
    st.error(f"CSV is missing required columns: {missing}")
    st.stop()

L_MIN, L_MAX = float(df["Lpatch_mm"].min()), float(df["Lpatch_mm"].max())
W_MIN, W_MAX = float(df["Wpatch_mm"].min()), float(df["Wpatch_mm"].max())

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
