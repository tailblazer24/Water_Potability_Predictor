import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page config

st.set_page_config(
    page_title="AquaGuard - Water Potability Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS - deep black + electric cyan

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>

/* Base */
.stApp {
    background-color: #080808;
    font-family: 'Syne', sans-serif;
    color: #e8e8e8;
}



/*  Main Header */
.main-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.2rem;
    letter-spacing: 7px;
    background: linear-gradient(90deg, #ffffff 0%, #00d4ff 55%, #0099bb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.main-subtitle {
    font-size: 0.78rem;
    color: #3a6070;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/*  Input card  */
.input-card {
    background: #0d0d0d;
    border: 1px solid #1a2e3a;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
}
.section-label {
    font-size: 0.62rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #00d4ff;
    font-weight: 700;
    margin-bottom: 1.2rem;
}

/*  Widget labels  */
label[data-testid="stWidgetLabel"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.74rem !important;
    color: #507080 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
input[type="number"] {
    background: #111111 !important;
    border: 1px solid #1a2e3a !important;
    color: #00d4ff !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 6px !important;
    font-size: 0.88rem !important;
}
input[type="number"]:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

/* Slider  */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #00d4ff !important;
    border-color: #00d4ff !important;
}
.stSlider > div > div > div > div {
    background: #00d4ff !important;
}

/*  Button  */
.stButton > button {
    background: linear-gradient(90deg, #00d4ff, #0099bb) !important;
    color: #000000 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.15rem !important;
    letter-spacing: 3px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.25) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(0,212,255,0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/*  Result cards */
.result-safe {
    background: linear-gradient(135deg, #001a0d, #002a14);
    border: 1px solid #00c853;
    border-left: 4px solid #00c853;
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin: 1rem 0;
}
.result-unsafe {
    background: linear-gradient(135deg, #1a0000, #2a0800);
    border: 1px solid #ff3d00;
    border-left: 4px solid #ff3d00;
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin: 1rem 0;
}
.result-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 3px;
    margin-bottom: 0.15rem;
}
.result-safe .result-label  { color: #00c853; }
.result-unsafe .result-label { color: #ff3d00; }
.result-prob {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #507080;
}
.result-safe   .result-prob span { color: #00c853; font-weight: 500; }
.result-unsafe .result-prob span { color: #ff3d00; font-weight: 500; }

/*  Tabs  */
.stTabs [data-baseweb="tab-list"] {
    background: #0a0a0a;
    border-bottom: 1px solid #1a2e3a;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.74rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #3a6070 !important;
    background: transparent !important;
    border: none !important;
    padding: 0.65rem 1.2rem !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #0a0a0a;
    border: 1px solid #1a2e3a;
    border-top: none;
    border-radius: 0 0 12px 12px;
    padding: 1.4rem;
}

/*  Caption */
.stCaptionContainer p, small {
    color: #3a6070 !important;
    font-size: 0.76rem !important;
    font-family: 'Syne', sans-serif !important;
}

/*  Divider */
hr { border-color: #1a2e3a !important; margin: 1.5rem 0 !important; }
            

/*  Hide chrome  */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Model loading

with open('water_quality_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

feature_order = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
                 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

dataset_means = {
    'Not Potable (avg)': {
        'ph': 7.078, 'Hardness': 196.733, 'Solids': 21777.491,
        'Chloramines': 7.092, 'Sulfate': 334.200, 'Conductivity': 426.730,
        'Organic_carbon': 14.364, 'Trihalomethanes': 66.321, 'Turbidity': 3.966
    },
    'Potable (avg)': {
        'ph': 7.069, 'Hardness': 195.801, 'Solids': 22383.991,
        'Chloramines': 7.169, 'Sulfate': 332.683, 'Conductivity': 425.384,
        'Organic_carbon': 14.161, 'Trihalomethanes': 66.543, 'Turbidity': 3.968
    }
}

# Dark chart style

def apply_dark_style(fig, ax):
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.tick_params(colors='#507080', labelsize=9)
    ax.xaxis.label.set_color('#507080')
    ax.yaxis.label.set_color('#507080')
    ax.title.set_color('#e0e0e0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2e3a')
    ax.grid(axis='x', color='#1a2e3a', linewidth=0.7, linestyle='--')
    ax.set_axisbelow(True)

# Main header

st.markdown('<div class="main-title">AQUASHIELD - WATER POTABILITY CALCULATOR </div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Enter parameters below </div>',
    unsafe_allow_html=True
)

# Input section

st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Parameters</div>', unsafe_allow_html=True)

inputs = {}
col1, col2 = st.columns(2, gap="large")
with col1:
    inputs['ph']             = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1)
    inputs['Hardness']       = st.number_input("Hardness (mg/L)", min_value=0.0, value=200.0)
    inputs['Solids']         = st.number_input("Total Dissolved Solids (ppm)", min_value=0.0, value=20000.0)
    inputs['Chloramines']    = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.0)
    inputs['Sulfate']        = st.number_input("Sulfate (mg/L)", min_value=0.0, value=300.0)
with col2:
    inputs['Conductivity']      = st.number_input("Conductivity (µS/cm)", min_value=0.0, value=400.0)
    inputs['Organic_carbon']    = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=15.0)
    inputs['Trihalomethanes']   = st.number_input("Trihalomethanes (µg/L)", min_value=0.0, value=70.0)
    inputs['Turbidity']         = st.number_input("Turbidity (NTU)", min_value=0.0, value=4.0)

st.markdown('</div>', unsafe_allow_html=True)

btn_col, _ = st.columns([1, 2])
with btn_col:
    predict = st.button("ANALYSE SAMPLE")

# Prediction & insights

if predict:
    input_df     = pd.DataFrame([inputs])[feature_order]
    scaled_input = scaler.transform(input_df)
    prediction   = model.predict(scaled_input)[0]
    probability  = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.markdown(
            f'<div class="result-safe">'
            f'<div class="result-label">✓ POTABLE — SAFE TO DRINK</div>'
            f'<div class="result-prob">Confidence: <span>{probability:.1%}</span></div>'
            f'</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-unsafe">'
            f'<div class="result-label">✕ NOT POTABLE — UNSAFE</div>'
            f'<div class="result-prob">Potability probability: <span>{probability:.1%}</span></div>'
            f'</div>', unsafe_allow_html=True
        )

    st.divider()
    st.markdown('<div class="section-label">Model Insights</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Global Feature Importance", "Sample vs. Dataset Averages"])

    #  Tab 1: Global Feature Importance 
    with tab1:
        st.caption(
            "Which features the Random Forest model weighs most heavily across all predictions - "
            "a global property of the trained model, not specific to the values you entered."
        )
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({'Feature': feature_order, 'Importance': importances})
        feat_imp = feat_imp.sort_values('Importance')

        n = len(feat_imp)
        bar_colors = [
            (0, (0.5 + 0.5 * i / (n - 1)) * 0.83, (0.5 + 0.5 * i / (n - 1)), 1.0)
            for i in range(n)
        ]

        fig1, ax1 = plt.subplots(figsize=(7, 4.2))
        apply_dark_style(fig1, ax1)
        ax1.barh(feat_imp['Feature'], feat_imp['Importance'],
                 color=bar_colors, height=0.58, edgecolor='none')
        ax1.set_xlabel("Mean Decrease in Impurity", fontsize=9)
        ax1.set_title("Global Feature Importance", fontsize=11, pad=12,
                      fontweight='600', color='#e0e0e0')
        ax1.tick_params(axis='y', labelsize=9, colors='#507080')
        fig1.tight_layout()
        st.pyplot(fig1)

    # Tab 2: Sample vs Dataset Averages 
    with tab2:
        st.caption(
            "Your entered values compared against the average of safe and unsafe water samples "
            "from the training data. Z-score normalised so all features share the same scale."
        )
        compare_df = pd.DataFrame(dataset_means).T
        compare_df.loc['Your Input'] = inputs
        normalised = compare_df.copy()
        for col in feature_order:
            col_mean = compare_df[col].mean()
            col_std  = compare_df[col].std()
            normalised[col] = (compare_df[col] - col_mean) / col_std if col_std > 0 else 0.0

        plot_data = normalised[feature_order].T.reset_index()
        plot_data.columns = ['Feature', 'Not Potable (avg)', 'Potable (avg)', 'Your Input']
        plot_melted = plot_data.melt(
            id_vars='Feature', var_name='Category', value_name='Normalised Value'
        )

        palette = {
            'Not Potable (avg)': '#ff4444',
            'Potable (avg)':     '#00c853',
            'Your Input':        '#00d4ff'
        }

        fig2, ax2 = plt.subplots(figsize=(9, 4.8))
        apply_dark_style(fig2, ax2)
        sns.barplot(data=plot_melted, x='Feature', y='Normalised Value',
                    hue='Category', ax=ax2, palette=palette, alpha=0.88)
        ax2.set_title("Your Sample vs. Dataset Averages (z-score normalised)",
                      fontsize=11, pad=12, fontweight='600', color='#e0e0e0')
        ax2.set_xlabel("")
        ax2.set_ylabel("Normalised Value (z-score)", fontsize=9)
        ax2.tick_params(axis='x', rotation=30, labelsize=8.5, colors='#507080')
        ax2.tick_params(axis='y', labelsize=8.5, colors='#507080')
        ax2.axhline(0, color='#1a2e3a', linewidth=1, linestyle='--')
        legend = ax2.legend(
            loc='upper right', framealpha=0.2,
            facecolor='#0d0d0d', edgecolor='#1a2e3a',
            labelcolor='#7ab8cc', fontsize=8
        )
        fig2.tight_layout()
        st.pyplot(fig2)
