import streamlit as st
import pandas as pd
import numpy as np
import keras
import joblib

# --- Configuration ---
FEATURE_COLS = ["gold_diff", "exp_diff", "kills_diff", "dragons_diff", "deaths_diff"]

# --- Load Models (Global Loading for Max Speed) ---

try:
    # 🔴 CHANGE 1: Load the new Keras Baseline MLP model
    BASELINE_MODEL = keras.models.load_model('baseline_mlp_model.keras')
    X_SCALER = joblib.load('x_scaler.joblib')

    DL_ENCODER = keras.models.load_model('dl_encoder.keras')
    DL_MLP_MODEL = keras.models.load_model('dl_mlp_model.keras')

except FileNotFoundError as e:
    st.error(
        f"Error: Model file not found. Please ensure all model files are in the same directory as app.py. Missing file: {e.filename}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during model loading. Check your Keras/TensorFlow installation. Error: {e}")
    st.stop()


# --- Prediction and Helper Functions (Use st.cache_data) ---

@st.cache_data
def get_scaler_stats(_scaler_obj):
    """
    Calculates min/max of training data range based on the Scaler object,
    then expands that range by 40% for wider slider inputs.
    """
    # 🔴 Define the expansion factor
    EXPANSION_FACTOR = 2.30  # 1.0 (original) + 0.40 (increase)

    n_features = _scaler_obj.n_features_in_

    # Define the new expanded scaled boundaries
    # Instead of -1.0 and 1.0, we use -1.4 and 1.4
    min_scaled = np.full((1, n_features), -EXPANSION_FACTOR)
    max_scaled = np.full((1, n_features), EXPANSION_FACTOR)

    # Inverse transform the new expanded scaled boundaries to get the new range
    min_orig = _scaler_obj.inverse_transform(min_scaled)[0]
    max_orig = _scaler_obj.inverse_transform(max_scaled)[0]

    return pd.DataFrame({
        'min': min_orig,
        'max': max_orig,
    }, index=FEATURE_COLS)


@st.cache_data
def predict_baseline(features):
    """
    🔴 CHANGE 2: Updated to predict using the Baseline MLP model.
    """
    features_scaled = X_SCALER.transform(np.array(features).reshape(1, -1))

    # MLP returns the probability directly (1 output unit, sigmoid activation)
    proba = BASELINE_MODEL.predict(features_scaled, verbose=0)[0][0]

    return float(proba)

@st.cache_data
def predict_dl(features):
    """Predicts using the Deep Learning (Autoencoder + MLP) model. Cached by input features."""
    features_scaled = X_SCALER.transform(np.array(features).reshape(1, -1))
    features_encoded = DL_ENCODER.predict(features_scaled, verbose=0)
    proba = DL_MLP_MODEL.predict(features_encoded, verbose=0)[0][1]
    return float(proba)


# --- Streamlit App Layout ---

st.set_page_config(
    page_title="LoL 10-Min Win Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("League of Legends 10-Minute Win Predictor 🧙‍♂️")
st.markdown("Use the sliders to input the **Blue Team's advantage** (Difference: Blue - Red) at the 10-minute mark and see the predicted probability of a **Blue Team Victory** (blueWins=1).")

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

# --- Main Content Tabs ---

tab_baseline, tab_dl = st.tabs(["📊 Baseline Model (Simple MLP)", "🧠 Deep Learning Model (Autoencoder + MLP)"])

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

# --- Baseline Tab Content ---
with tab_baseline:
    st.header("Baseline Model Prediction")
    # 🔴 CHANGE 4: Update text description
    st.markdown("This prediction uses a simple **Multi-Layer Perceptron (MLP)** model trained directly on the 5 scaled features.")

    col_input, col_pred = st.columns(2)

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

# --- Deep Learning Tab Content ---
with tab_dl:
    st.header("Deep Learning Model Prediction")
    st.markdown(
        "This prediction uses a **Multi-Layer Perceptron (MLP)** trained on features encoded by an **Autoencoder**.")

    col_input_dl, col_pred_dl = st.columns(2)

    with col_input_dl:
        st.subheader("Current Input")
        st.table(df_input_display)

    with col_pred_dl:
        # Get prediction
        proba_dl = predict_dl(input_features)

        st.subheader("Predicted Blue Win Probability")

        st.metric(label="Blue Team Win Chance", value=f"{proba_dl * 100:.2f}%")
        st.progress(proba_dl)

        if proba_dl > 0.5:
            st.success("Prediction: Blue Team Victory! 🟦")
        elif proba_dl < 0.5:
            st.error("Prediction: Red Team Victory! 🟥")
        else:
            st.warning("Prediction: Too close to call! 🟨")

st.markdown(
    """
    ***
    ### Model Details:
    - **Data**: High-Diamond Ranked 10-Minute Game Data.
    - **Features**: Differences (Blue - Red) in Gold, Experience, Kills, Dragons, and Deaths.
    - **Baseline Model**: **Simple MLP** (1 Hidden Layer, ReLU activation, 8 neurons).
    - **DL Model**: Autoencoder (Tanh, 8-dim) $\\rightarrow$ MLP (2 Dense layers, Tanh, BN) $\\rightarrow$ Output (Sigmoid).
    """
)