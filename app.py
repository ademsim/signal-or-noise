"""signal-or-noise: a fixed-threshold rule vs a machine-learning model under added noise.

Toy demo on public neutron/gamma pulses. The noise is synthetic (Gaussian), so the
results illustrate an idea; they say nothing about any specific real detector system.

Run:  streamlit run app.py
Needs: models/model.pkl  (dict with keys "model" and "thr")
       sample/test_sample.npz  (arrays "X" with shape (n, 280) and "y" with 0=gamma, 1=neutron)
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "model.pkl"
SAMPLE_PATH = BASE / "sample" / "test_sample.npz"
TAIL_START = 80          # the tail is everything from sample 80 on
SIGMA_MAX = 0.3
CLASS_NAMES = {0: "gamma", 1: "neutron"}

st.set_page_config(page_title="signal-or-noise", layout="wide")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)          # only unpickle files you created yourself
    return artifact["model"], float(artifact["thr"])


@st.cache_data
def load_sample():
    data = np.load(SAMPLE_PATH)
    return data["X"].astype(np.float32), data["y"].astype(int)


def tail_to_total(X):
    """Share of the pulse area that lies in the tail."""
    return X[:, TAIL_START:].sum(axis=1) / X.sum(axis=1)


def add_noise(X, sigma, seed):
    rng = np.random.default_rng(seed)
    return X + rng.normal(0.0, sigma, size=X.shape)


def accuracy(pred, y):
    return float((pred == y).mean())


def rates(pred, y):
    """TPR: share of real neutrons flagged. FPR: share of real gammas wrongly flagged."""
    return float(pred[y == 1].mean()), float(pred[y == 0].mean())


@st.cache_data
def accuracy_curve(_model, thr, seed, X, y):
    """Accuracy of both methods over a grid of noise levels (fixed seed)."""
    sigmas = np.linspace(0.0, SIGMA_MAX, 13)
    rule, ml = [], []
    for s in sigmas:
        Xn = add_noise(X, s, seed)
        rule.append(accuracy((tail_to_total(Xn) > thr).astype(int), y))
        ml.append(accuracy(_model.predict(Xn), y))
    return sigmas, rule, ml


# ---------------------------------------------------------------- missing files
if not (MODEL_PATH.exists() and SAMPLE_PATH.exists()):
    st.error(
        "Missing files. Create them in the Colab notebook and place them here:\n\n"
        f"- `{MODEL_PATH.relative_to(BASE)}`\n- `{SAMPLE_PATH.relative_to(BASE)}`"
    )
    st.stop()

model, thr_train = load_model()
X, y = load_sample()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Controls")
    sigma = st.slider("Noise level (sigma)", 0.0, SIGMA_MAX, 0.05, 0.01)
    thr = st.slider("Rule threshold (tail/total)", 0.25, 0.45, round(thr_train, 3), 0.001)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=0, step=1)
    event = st.slider("Example event", 0, len(X) - 1, 0)
    st.caption(f"Threshold learned from training data: {thr_train:.4f}")

# ---------------------------------------------------------------- compute
Xn = add_noise(X, sigma, seed)
ratio = tail_to_total(Xn)
pred_rule = (ratio > thr).astype(int)
pred_ml = model.predict(Xn).astype(int)
acc_rule, acc_ml = accuracy(pred_rule, y), accuracy(pred_ml, y)
acc_lazy = accuracy(np.zeros_like(y), y)       # "always gamma"

# ---------------------------------------------------------------- page
st.title("signal-or-noise")
st.write(
    "Should an event count as **signal** (neutron) or **noise** (gamma)? "
    "Compare a single hand-tuned threshold with a model that learned from data, "
    "while you turn up the noise."
)
st.caption(f"{len(X)} test pulses, {int(y.sum())} neutrons. Noise is synthetic Gaussian.")

c1, c2, c3 = st.columns(3)
c1.metric("Rule (tail/total)", f"{acc_rule:.3f}")
c2.metric("Logistic regression", f"{acc_ml:.3f}")
c3.metric("Always 'gamma'", f"{acc_lazy:.3f}")

tpr_r, fpr_r = rates(pred_rule, y)
tpr_m, fpr_m = rates(pred_ml, y)
st.table(
    pd.DataFrame(
        {"TPR (neutrons caught)": [tpr_r, tpr_m], "FPR (gammas flagged)": [fpr_r, fpr_m]},
        index=["Rule", "Logistic regression"],
    ).round(4)
)

left, right = st.columns(2)

with left:
    st.subheader("Where the rule draws its line")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    lo, hi = np.percentile(ratio, [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    ax.hist(ratio[y == 0], bins=bins, density=True, alpha=0.6, color="tab:blue", label="gamma")
    ax.hist(ratio[y == 1], bins=bins, density=True, alpha=0.6, color="tab:red", label="neutron")
    ax.axvline(thr, color="black", linestyle="--", label="threshold")
    ax.set_xlabel("tail / total")
    ax.set_ylabel("density")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader(f"Example event ({CLASS_NAMES[int(y[event])]})")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(X[event], color="gray", alpha=0.8, label="clean")
    ax.plot(Xn[event], color="tab:orange", alpha=0.8, label="with noise")
    ax.axvline(TAIL_START, color="black", linestyle=":", label="tail starts")
    ax.set_xlabel("time (sample index)")
    ax.set_ylabel("amplitude")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f"Rule says: {CLASS_NAMES[int(pred_rule[event])]}. "
        f"Model says: {CLASS_NAMES[int(pred_ml[event])]}."
    )

st.subheader("Accuracy as noise grows")
sigmas, rule_curve, ml_curve = accuracy_curve(model, thr, int(seed), X, y)
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(sigmas, rule_curve, marker="o", label="rule (tail/total)")
ax.plot(sigmas, ml_curve, marker="o", label="logistic regression")
ax.axhline(acc_lazy, color="gray", linestyle="--", label="always 'gamma'")
ax.axvline(sigma, color="black", linestyle=":", label="current sigma")
ax.set_xlabel("sigma (noise level)")
ax.set_ylabel("accuracy")
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.divider()
st.caption(
    "Data: Liu et al., 'Pulse Shape Discrimination Algorithms: A Survey and Benchmark', "
    "Zenodo, doi:10.5281/zenodo.16730902 (EJ299-33 scintillator, AmBe source). "
    "Both methods are trained on clean training data only; noise is added to the test sample."
)
