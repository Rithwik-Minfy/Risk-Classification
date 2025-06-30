import os
import streamlit as st
import pandas as pd
import joblib

# ─── CACHING ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts(models_dir='saved_models'):
    pt       = joblib.load(os.path.join(models_dir, 'pt.pkl'))
    rs       = joblib.load(os.path.join(models_dir, 'rs.pkl'))
    ss       = joblib.load(os.path.join(models_dir, 'ss.pkl'))
    selector = joblib.load(os.path.join(models_dir, 'selector.pkl'))

    models = {}
    for fn in os.listdir(models_dir):
        if fn.endswith('_model.pkl'):
            name = fn.replace('_model.pkl', '')
            models[name] = joblib.load(os.path.join(models_dir, fn))

    return pt, rs, ss, selector, models

def preprocess_and_select(df, pt, rs, ss, selector):
    df = df.drop(['ID', 'ZIP Code', 'Personal Loan'], axis=1, errors='ignore')
    rb_cols  = ['CCAvg', 'Mortgage']
    std_cols = ['Income', 'Experience', 'Age']

    df[rb_cols]  = pt.transform(df[rb_cols])
    df[rb_cols]  = rs.transform(df[rb_cols])
    df[std_cols] = ss.transform(df[std_cols])

    X_sel = selector.transform(df)
    sel_cols = [f'F{i}' for i in range(X_sel.shape[1])]
    return pd.DataFrame(X_sel, columns=sel_cols)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏦 Bank Loan Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM STYLE ─────────────────────────────────────────────────────────
st.markdown("""
    <style>
        body { background-color: #f0f2f6; }
        .reportview-container .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 {
            color: #0d6efd;
        }
        .stButton>button {
            background-color: #0d6efd;
            color: white;
            border-radius: 0.5rem;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #0b5ed7;
            transform: scale(1.02);
        }
        .stDownloadButton>button {
            background-color: #20c997;
            color: white;
            border-radius: 0.5rem;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            transition: all 0.2s ease-in-out;
        }
        .stDownloadButton>button:hover {
            background-color: #198754;
            transform: scale(1.02);
        }
    </style>
""", unsafe_allow_html=True)

# ─── TITLE & INTRO ────────────────────────────────────────────────────────
st.title("🏦 Bank Loan Risk Classification System")

st.markdown("""
<div style="text-align:center; font-size:1.2rem; margin-bottom:1rem;">
    Upload your customer CSV file to get predictions from multiple ML models.
</div>
""", unsafe_allow_html=True)

# ─── LOAD MODELS ──────────────────────────────────────────────────────────
pt, rs, ss, selector, models = load_artifacts()

# ─── FILE UPLOAD ──────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📁 Upload CSV File",
    type=['csv'],
    help="Ensure it has features like Income, Age, Experience, CCAvg, Mortgage."
)

# ─── PREDICTION PIPELINE ──────────────────────────────────────────────────
if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"❌ Could not read CSV: {e}")
    else:
        st.subheader("👀 Preview of Uploaded Data")
        st.dataframe(df.head(), use_container_width=True)

        with st.spinner("🔄 Preprocessing and running predictions..."):
            X_sel = preprocess_and_select(df.copy(), pt, rs, ss, selector)
            for name, model in models.items():
                df[f'Pred_{name}'] = model.predict(X_sel)

        st.success("✅ Predictions complete!")

        st.subheader("📊 Results with Predictions")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
else:
    st.info("🔽 Upload a file to begin. Only CSVs are supported.")
