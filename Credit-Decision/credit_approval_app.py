"""
Credit and Loan Approval Decision Explorer
--------------------------------------------
An interactive Streamlit app demonstrating how four underwriting inputs feed
a behind-the-scenes credit decision, compared across four modelling
approaches (Logistic Regression, Random Forest, a scikit-learn neural
network, and a PyTorch neural network with a graceful fallback if PyTorch
is not installed).

All data here is synthetic, generated from a known approval process, purely
to illustrate how the models behave, not a real underwriting system.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# PyTorch is optional. If it isn't installed, the app still runs, and the
# "Neural Network (PyTorch)" option falls back to a clearly-labeled
# scikit-learn equivalent instead of crashing.
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    pass

st.set_page_config(page_title="Credit & Loan Approval Explorer", layout="wide")

RATING_MAP = {"AAA": 8, "AA": 7, "A": 6, "BBB": 5, "BB": 4, "B": 3, "CCC": 2, "CC / C": 1}
FEATURE_NAMES = ["Credit Rating Score", "Collateral Coverage Ratio", "Credit Duration (yrs)", "Net Tangible Worth ($M)"]

APPROVE_COLOR = "#1B9E77"
REJECT_COLOR = "#D95F02"
DECISION_CMAP = LinearSegmentedColormap.from_list("decision", [REJECT_COLOR, "#F5F5DC", APPROVE_COLOR])


# ---------------------------------------------------------------------------
# Synthetic data and model training (cached so the app doesn't retrain on
# every widget interaction, only the first time, or if the code changes)
# ---------------------------------------------------------------------------

@st.cache_data
def generate_training_data(n=3000, seed=42):
    """
    Simulates a loan book with a KNOWN underlying approval process, so the
    models below can be checked against a genuine, if synthetic, ground
    truth rather than an arbitrary black box.
    """
    rng = np.random.default_rng(seed)
    credit_score = rng.integers(1, 9, n).astype(float)        # 1 (CC/C) to 8 (AAA)
    collateral_ratio = rng.uniform(0.0, 3.0, n)                # collateral value / loan amount
    duration = rng.uniform(1, 30, n)                           # loan term, years
    net_worth = rng.uniform(-20, 150, n)                       # tangible assets + cash - long-term debt, $M

    logit = (
        -4.6
        + 0.62 * credit_score
        + 1.15 * collateral_ratio
        - 0.045 * duration
        + 0.028 * net_worth
        - 0.05 * np.maximum(4 - credit_score, 0) * (duration / 10)
    )
    prob = 1 / (1 + np.exp(-logit))
    approved = (rng.uniform(0, 1, n) < prob).astype(int)

    X = np.column_stack([credit_score, collateral_ratio, duration, net_worth])
    return X, approved


class TorchNet(nn.Module if TORCH_AVAILABLE else object):
    """A small feedforward network, only defined if PyTorch is available."""
    def __init__(self, n_features, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@st.cache_resource
def train_models():
    X, y = generate_training_data()
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)

    logreg = LogisticRegression(max_iter=1000).fit(X_s, y)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0).fit(X, y)
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", alpha=0.01,
                         max_iter=2000, random_state=0).fit(X_s, y)

    torch_model = None
    if TORCH_AVAILABLE:
        torch_model = TorchNet(n_features=4)
        optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01, weight_decay=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        X_t = torch.tensor(X_s, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        for _ in range(300):
            optimizer.zero_grad()
            logits = torch_model(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            optimizer.step()
        torch_model.eval()

    return {"scaler": scaler, "logreg": logreg, "rf": rf, "mlp": mlp, "torch_model": torch_model}


def predict_all_models(models, x_raw):
    """Returns approval probability from every model for a single applicant."""
    x_raw = np.array(x_raw, dtype=float).reshape(1, -1)
    x_s = models["scaler"].transform(x_raw)

    probs = {
        "Logistic Regression": models["logreg"].predict_proba(x_s)[0, 1],
        "Random Forest": models["rf"].predict_proba(x_raw)[0, 1],
        "Neural Network (scikit-learn)": models["mlp"].predict_proba(x_s)[0, 1],
    }
    if TORCH_AVAILABLE and models["torch_model"] is not None:
        with torch.no_grad():
            logit = models["torch_model"](torch.tensor(x_s, dtype=torch.float32)).item()
        probs["Neural Network (PyTorch)"] = 1 / (1 + np.exp(-logit))
    else:
        # graceful fallback: reuse the scikit-learn network's prediction, clearly labeled,
        # so the app and its dropdown still work end to end without PyTorch installed
        probs["Neural Network (PyTorch)"] = probs["Neural Network (scikit-learn)"]
    return probs


def predict_grid(models, model_name, duration_fixed, net_worth_fixed, n_grid=70):
    """Approval probability across a credit-score x collateral-ratio grid, holding the
    other two inputs fixed at their current sidebar values, for the decision-region plot."""
    cs_grid = np.linspace(1, 8, n_grid)
    cr_grid = np.linspace(0, 3, n_grid)
    CS, CR = np.meshgrid(cs_grid, cr_grid)
    flat_X = np.column_stack([
        CS.ravel(), CR.ravel(),
        np.full(CS.size, duration_fixed), np.full(CS.size, net_worth_fixed)
    ])
    flat_X_s = models["scaler"].transform(flat_X)

    if model_name == "Logistic Regression":
        probs = models["logreg"].predict_proba(flat_X_s)[:, 1]
    elif model_name == "Random Forest":
        probs = models["rf"].predict_proba(flat_X)[:, 1]
    elif model_name == "Neural Network (scikit-learn)":
        probs = models["mlp"].predict_proba(flat_X_s)[:, 1]
    elif model_name == "Neural Network (PyTorch)":
        if TORCH_AVAILABLE and models["torch_model"] is not None:
            with torch.no_grad():
                logits = models["torch_model"](torch.tensor(flat_X_s, dtype=torch.float32)).numpy()
            probs = 1 / (1 + np.exp(-logits))
        else:
            probs = models["mlp"].predict_proba(flat_X_s)[:, 1]
    return CS, CR, probs.reshape(CS.shape)


# ---------------------------------------------------------------------------
# Sidebar: the four underwriting inputs, plus the model-choice dropdown
# ---------------------------------------------------------------------------

st.sidebar.header("Applicant Profile")

rating_label = st.sidebar.selectbox("Institution Credit Rating", list(RATING_MAP.keys()), index=3)
credit_score = RATING_MAP[rating_label]

collateral_ratio = st.sidebar.slider(
    "Estimated Collateral Coverage Ratio", min_value=0.0, max_value=3.0, value=1.0, step=0.05,
    help="Collateral value as a multiple of the loan amount (1.0 = fully collateralized)."
)

duration = st.sidebar.slider(
    "Duration of Credit (years)", min_value=1, max_value=30, value=10, step=1
)

net_worth = st.sidebar.slider(
    "Tangible Assets + Cash − Long-Term Debt ($M)", min_value=-20.0, max_value=150.0, value=25.0, step=1.0,
    help="Net tangible liquid worth: tangible assets plus cash, less long-term debt."
)

st.sidebar.markdown("---")
st.sidebar.header("Modelling Approach")

model_options = ["Logistic Regression", "Random Forest", "Neural Network (scikit-learn)", "Neural Network (PyTorch)"]
selected_model = st.sidebar.selectbox("Model used for the decision region below", model_options, index=0)

if not TORCH_AVAILABLE:
    st.sidebar.caption(
        "PyTorch is not installed in this environment; the PyTorch option falls back to the "
        "scikit-learn neural network so the app still runs end to end."
    )

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("Credit and Loan Approval Decision Explorer")
st.caption(
    "A synthetic demonstration: four underwriting inputs feed a behind-the-scenes approval "
    "decision, compared across four modelling approaches. All data and outcomes here are simulated."
)

models = train_models()
current_x = [credit_score, collateral_ratio, duration, net_worth]
all_probs = predict_all_models(models, current_x)
selected_prob = all_probs[selected_model]
decision = "APPROVED" if selected_prob >= 0.5 else "REJECTED"
decision_color = APPROVE_COLOR if decision == "APPROVED" else REJECT_COLOR

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Current Decision")
    st.markdown(
        f"<div style='padding:1.2em; border-radius:10px; background-color:{decision_color}; "
        f"color:white; text-align:center; font-size:1.4em; font-weight:bold;'>{decision}</div>",
        unsafe_allow_html=True,
    )
    st.metric(f"Approval probability ({selected_model})", f"{selected_prob:.1%}")

    st.markdown("##### Applicant summary")
    st.table(pd.DataFrame({
        "Input": ["Credit Rating", "Collateral Coverage", "Duration (yrs)", "Net Tangible Worth ($M)"],
        "Value": [str(rating_label), f"{collateral_ratio:.2f}x", str(duration), f"{net_worth:,.1f}"],
    }).set_index("Input"))

with col2:
    st.subheader("Approval Probability by Model")
    st.caption("Every model sees the same applicant. They should broadly agree on the decision, but rarely agree exactly.")
    fig_bar, ax_bar = plt.subplots(figsize=(7, 3.8))
    model_names = list(all_probs.keys())
    prob_values = [all_probs[m] for m in model_names]
    bar_colors = [APPROVE_COLOR if p >= 0.5 else REJECT_COLOR for p in prob_values]
    ax_bar.barh(model_names, prob_values, color=bar_colors)
    ax_bar.axvline(0.5, color="#333333", linestyle="--", linewidth=1, label="Decision threshold")
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Approval probability")
    ax_bar.legend(loc="lower right", fontsize=8)
    for i, p in enumerate(prob_values):
        ax_bar.text(p + 0.02 if p < 0.9 else p - 0.1, i, f"{p:.1%}", va="center",
                    color="black" if p < 0.9 else "white", fontsize=9)
    st.pyplot(fig_bar)

st.markdown("---")
st.subheader(f"Decision Region: {selected_model}")
st.caption(
    "Credit rating vs. collateral coverage, with duration and net tangible worth held at the sidebar "
    "values above. Green shading marks combinations this model would approve; orange marks rejection. "
    "The current applicant is plotted as a white star."
)

CS, CR, prob_grid = predict_grid(models, selected_model, duration, net_worth)

fig_grid, ax_grid = plt.subplots(figsize=(9, 6))
contour = ax_grid.contourf(CS, CR, prob_grid, levels=np.linspace(0, 1, 21), cmap=DECISION_CMAP)
ax_grid.contour(CS, CR, prob_grid, levels=[0.5], colors="black", linewidths=1.8, linestyles="--")
ax_grid.scatter([credit_score], [collateral_ratio], color="white", edgecolor="black",
                 s=260, marker="*", zorder=5, label="Current applicant")
ax_grid.set_xlabel("Credit Rating Score (1 = CC/C, 8 = AAA)")
ax_grid.set_ylabel("Collateral Coverage Ratio")
ax_grid.legend(loc="upper left")
cbar = fig_grid.colorbar(contour, ax=ax_grid)
cbar.set_label("Approval probability")
st.pyplot(fig_grid)

with st.expander("How this demo works"):
    st.markdown(
        """
        A synthetic loan book of 3,000 simulated applicants is generated from a known approval
        process (a logistic function of credit rating, collateral coverage, duration, and net
        tangible worth, with one deliberate interaction: weak credit combined with a long duration
        is penalized more than either factor alone). Four models are then trained on that same
        data: Logistic Regression, a Random Forest, a scikit-learn neural network, and a PyTorch
        neural network. Because all four see the same training data and the same true underlying
        pattern, their predictions broadly agree, but each model's own decision boundary, and its
        specific probability estimate for a borderline applicant, will differ at the margin. That
        divergence is most visible for applicants who land close to the dashed 50% boundary in the
        decision region plot above; strong or weak applicants tend to be approved or rejected by
        every model in agreement.
        """
    )
