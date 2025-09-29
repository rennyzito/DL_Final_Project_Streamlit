import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration ---
FEATURE_COLS = ["gold_diff", "exp_diff", "kills_diff", "dragons_diff", "deaths_diff"]

# --- Load Models (Baseline Only) ---
# NOTE: We skip loading Keras/TensorFlow to test the speed difference.
try:
    # Load scikit-learn objects (Logistic Regression and StandardScaler)
    BASELINE_MODEL = joblib.load('logistic_regression_model.joblib')
    X_SCALER = joblib.load('x_scaler.joblib')

except FileNotFoundError as e:
    st.error(
        f"Error: Model file not found. Please ensure all model files are in the same directory as this script. Missing file: {e.filename}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during baseline model loading. Error: {e}")
    st.stop()


# --- Prediction and Helper Functions ---

@st.cache_data
def get_scaler_stats(scaler_obj):
    """Calculates min/max of training data from the Scaler object to set slider limits."""
    n_features = scaler_obj.n_features_in_
    min_scaled = np.full((1, n_features), -1.0)
    max_scaled = np.full((1, n_features), 1.0)
    min_orig = scaler_obj.inverse_transform(min_scaled)[0]
    max_orig = scaler_obj.inverse_transform(max_scaled)[0]
    return pd.DataFrame({'min': min_orig, 'max': max_orig}, index=FEATURE_COLS)


@st.cache_data
def predict_baseline(features):
    """Predicts using the Baseline (Logistic Regression) model. Cached by input features."""
    features_scaled = X_SCALER.transform(np.array(features).reshape(1, -1))
    proba = BASELINE_MODEL.predict_proba(features_scaled)[0][1]
    return proba


# --- Streamlit App Layout ---

st.set_page_config(
    page_title="LoL Baseline Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("League of Legends 10-Minute Win Predictor (Baseline Only) 🚀")
st.markdown("This diagnostic version loads *only* the quick Baseline Model.")

# --- Sidebar for Feature Inputs ---

df_stats = get_scaler_stats(X_SCALER)

with st.sidebar:
    st.header("Input Features (Blue - Red)")

    # Sliders for user input
    gold_diff = st.slider(
        "Gold Difference (gold_diff)",
        min_value=int(df_stats.loc['gold_diff', 'min'] // 100 * 100),
        max_value=int(df_stats.loc['gold_diff', 'max'] // 100 * 100) + 100,
        value=0,
        step=50
    )
    # ... (other sliders omitted for brevity, but include them all) ...
    exp_diff = st.slider(
        "Experience Difference (exp_diff)",
        min_value=int(df_stats.loc['exp_diff', 'min'] // 100 * 100),
        max_value=int(df_stats.loc['exp_diff', 'max'] // 100 * 100) + 100,
        value=0,
        step=50
    )
    kills_diff = st.slider(
        "Kill Difference (kills_diff)",
        min_value=int(df_stats.loc['kills_diff', 'min'] - 1),
        max_value=int(df_stats.loc['kills_diff', 'max'] + 1),
        value=0,
        step=1
    )
    dragons_diff = st.slider(
        "Dragons Difference (dragons_diff)",
        min_value=int(df_stats.loc['dragons_diff', 'min'] - 1),
        max_value=int(df_stats.loc['dragons_diff', 'max'] + 1),
        value=0,
        step=1
    )
    deaths_diff = st.slider(
        "Deaths Difference (deaths_diff)",
        min_value=int(df_stats.loc['deaths_diff', 'min'] - 1),
        max_value=int(df_stats.loc['deaths_diff', 'max'] + 1),
        value=0,
        step=1
    )

    input_features = [gold_diff, exp_diff, kills_diff, dragons_diff, deaths_diff]

# --- Main Content ---
st.header("Baseline Model Prediction")

col_input, col_pred = st.columns(2)

# Prepare input table for display
df_input = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Value": input_features
})
df_input.loc[df_input['Feature'] == 'gold_diff', 'Feature'] = 'Gold Difference'
df_input.loc[df_input['Feature'] == 'exp_diff', 'Feature'] = 'Experience Difference'
df_input.loc[df_input['Feature'] == 'kills_diff', 'Feature'] = 'Kill Difference'
df_input.loc[df_input['Feature'] == 'dragons_diff', 'Feature'] = 'Dragon Difference'
df_input.loc[df_input['Feature'] == 'deaths_diff', 'Feature'] = 'Deaths Difference'
df_input_display = df_input.set_index('Feature')

with col_input:
    st.subheader("Current Input")
    st.table(df_input_display)

with col_pred:
    # Get prediction
    proba_baseline = predict_baseline(input_features)

    st.subheader("Predicted Blue Win Probability")
    st.metric(label="Blue Team Win Chance", value=f"{proba_baseline * 100:.2f}%")
    st.progress(proba_baseline)

    if proba_baseline > 0.5:
        st.success("Prediction: Blue Team Victory! 🟦")
    elif proba_baseline < 0.5:
        st.error("Prediction: Red Team Victory! 🟥")
    else:
        st.warning("Prediction: Too close to call! 🟨")

st.markdown(
    """
    ***
    ### Model Details:
    - **Baseline Model**: Logistic Regression. (No deep learning libraries were loaded for this test.)
    """
)