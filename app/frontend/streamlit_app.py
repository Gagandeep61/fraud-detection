import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# ── Load test data for random sampling ───────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
@st.cache_data
def load_test_data():
    # Load original data for random sample testing
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'creditcard.csv'))
    return df

@st.cache_data  
def load_comparison():
    try:
        comp = joblib.load(os.path.join(BASE_DIR, 'models', 'model_comparison.pkl'))
        return comp
    except:
        return None

df = load_test_data()
comparison_df = load_comparison()

API_URL = os.getenv("API_URL", "http://localhost:5000/predict")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Credit Card Fraud Detection System")
st.markdown("""
This system uses an **XGBoost model** trained on 284,000+ real transactions 
to detect fraudulent activity in real time.
""")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Model Settings")

# Threshold slider — the feature that makes this demo special
custom_threshold = st.sidebar.slider(
    "Detection Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Controls tradeoff between catching fraud and false alarms"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Understanding the Threshold

**Lower threshold →**
- Catches MORE fraud ✅
- More false alarms ❌
- Customers get declined more often

**Higher threshold →**
- Fewer false alarms ✅  
- Misses MORE fraud ❌
- Fraudsters slip through

**This is the core business tradeoff** in every real fraud system.
The optimal threshold was tuned using the Precision-Recall curve.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
st.sidebar.metric("Algorithm", "XGBoost")
st.sidebar.metric("ROC-AUC", "0.9788")
st.sidebar.metric("F1 Score", "0.8409")
st.sidebar.metric("Precision", "91.4%")

# ── Main Content: Three Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🎲 Test with Real Data",
    "📊 Model Comparison",
    "ℹ️ About This Project"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Test with Real Data
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Test the Model with Real Transactions")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Load a Transaction")
        
        sample_type = st.radio(
            "Choose sample type:",
            ["Random Transaction", "Random FRAUD Transaction", 
             "Random LEGITIMATE Transaction"]
        )
        
        if st.button("🎲 Load Sample Transaction", use_container_width=True):
            if sample_type == "Random FRAUD Transaction":
                sample = df[df['Class'] == 1].sample(1, random_state=None).iloc[0]
            elif sample_type == "Random LEGITIMATE Transaction":
                sample = df[df['Class'] == 0].sample(1, random_state=None).iloc[0]
            else:
                sample = df.sample(1, random_state=None).iloc[0]
            
            st.session_state['sample'] = sample
            st.session_state['true_label'] = int(sample['Class'])
        
        if 'sample' in st.session_state:
            true_label = st.session_state['true_label']
            if true_label == 1:
                st.error("**True Label: 🚨 FRAUD**")
            else:
                st.success("**True Label: ✅ LEGITIMATE**")
            
            st.caption("Transaction loaded. Click Analyze to see prediction.")
    
    with col2:
        st.subheader("Transaction Amount")
        if 'sample' in st.session_state:
            amount = st.session_state['sample']['Amount']
            time = st.session_state['sample']['Time']
            st.metric("Amount", f"${amount:.2f}")
            st.metric("Time", f"{time:.0f} seconds")
        else:
            st.info("Load a sample transaction to see details")
    
    st.markdown("---")
    
    # Analyze button
    if st.button("🔍 Analyze Transaction", 
                 use_container_width=True,
                 type="primary"):
        
        if 'sample' not in st.session_state:
            st.warning("Please load a sample transaction first!")
        else:
            sample = st.session_state['sample']
            
            # Build payload — all V features + Amount + Time
            payload = {
                "Time": float(sample['Time']),
                "Amount": float(sample['Amount'])
            }
            for i in range(1, 29):
                payload[f'V{i}'] = float(sample[f'V{i}'])
            
            with st.spinner("Analyzing transaction..."):
                try:
                    response = requests.post(API_URL, json=payload, timeout=10)
                    result = response.json()
                    
                    # Override threshold with slider value
                    fraud_prob = result['fraud_probability']
                    prediction = "fraud" if fraud_prob >= custom_threshold else "legitimate"
                    
                    st.markdown("---")
                    st.subheader("🤖 Model Prediction")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        if prediction == "fraud":
                            st.error("### 🚨 FRAUD DETECTED")
                        else:
                            st.success("### ✅ LEGITIMATE")
                    
                    with col_b:
                        st.metric(
                            "Fraud Probability",
                            f"{fraud_prob*100:.2f}%"
                        )
                    
                    with col_c:
                        st.metric(
                            "Risk Level",
                            result['risk_level']
                        )
                    
                    # Probability gauge
                    st.progress(fraud_prob)
                    st.caption(f"Fraud probability bar (threshold = {custom_threshold})")
                    
                    # Correct/Wrong indicator
                    if 'true_label' in st.session_state:
                        true = st.session_state['true_label']
                        pred_int = 1 if prediction == "fraud" else 0
                        
                        st.markdown("---")
                        if true == pred_int:
                            st.success(f"✅ **CORRECT PREDICTION** — Model correctly identified this as {'fraud' if true==1 else 'legitimate'}")
                        else:
                            if true == 1 and pred_int == 0:
                                st.error("❌ **FALSE NEGATIVE** — This was actual fraud but model missed it (missed fraud = money lost)")
                            else:
                                st.warning("⚠️ **FALSE POSITIVE** — Legitimate transaction flagged as fraud (false alarm = unhappy customer)")
                    
                    # Threshold insight
                    st.info(f"""
                    **Threshold Analysis:** 
                    Fraud probability is **{fraud_prob*100:.2f}%**.
                    At threshold **{custom_threshold}**, this is classified as **{prediction}**.
                    Try moving the threshold slider to see how the decision changes.
                    """)
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to the prediction API.")
                    st.info("""
                    **If running locally:** Start the Flask API first:
                    python app/api/app.py
                    **If this is the deployed app:** The API server may be 
                    sleeping (free tier). Wait 30 seconds and try again.
                    """)
    
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. The server is taking too long.")
                    st.info("Free tier servers sleep after inactivity. Wait 30 seconds and retry.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Model Comparison")
    st.markdown("Results from training 5 different approaches on this dataset.")
    
    if comparison_df is not None:
        # Display comparison table
        display_cols = ['model', 'f1', 'roc_auc', 'precision', 
                       'recall', 'true_positives', 'false_negatives', 
                       'false_positives']
        st.dataframe(
            comparison_df[display_cols].style.highlight_max(
                subset=['f1', 'roc_auc'],
                color='lightgreen'
            ).highlight_min(
                subset=['false_negatives', 'false_positives'],
                color='lightgreen'
            ),
            use_container_width=True
        )
    else:
        # Hardcode results if pkl not available
        data = {
            'Model': ['LR Baseline', 'LR Class Weights', 
                     'RF SMOTE', 'RF Undersampling', 'XGBoost'],
            'F1': [0.70, 0.10, 0.35, 0.17, 0.84],
            'ROC-AUC': [0.9561, 0.9636, 0.9726, 0.9694, 0.9788],
            'Recall': [0.59, 0.87, 0.83, 0.87, 0.78],
            'False Negatives': [39, 12, 16, 12, 21],
            'False Positives': [9, 1404, 280, 807, 7]
        }
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Key Insight — Why These Results Matter")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.error("""
        **The Accuracy Trap**
        
        A model predicting ALL transactions as legitimate 
        achieves **99.83% accuracy**.
        
        That model catches **ZERO fraud**.
        
        This is why we use F1, Precision, Recall, 
        and ROC-AUC instead.
        """)
    
    with col2:
        st.warning("""
        **The Core Business Tradeoff**
        
        **LR Class Weights**: Catches 83/95 frauds ✅
        But generates 1,404 false alarms ❌
        
        **XGBoost**: Only 7 false alarms ✅  
        But misses 21 frauds ❌
        
        Right choice depends on business priorities.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — About
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## Fraud Detection System
    
    ### Problem Statement
    Credit card fraud costs billions annually. Traditional rule-based systems 
    miss sophisticated fraud patterns. This ML system learns from 284,000+ 
    historical transactions to detect fraud automatically.
    
    ### The Core Challenge — Class Imbalance
    Only **0.17%** of transactions in this dataset are fraudulent. 
    This makes standard accuracy meaningless and requires specialized techniques.
    
    ### What Makes This Project Different
    
    **1. Rigorous EDA**
    - KS-Test statistical validation of feature importance  
    - KDE overlap analysis showing feature separability
    - t-SNE dimensionality reduction proving problem solvability
    - Cyclical time encoding (23:00 is close to 01:00)
    
    **2. Multiple Imbalance Techniques Compared**
    - Class Weights (free, always try first)
    - SMOTE (synthetic fraud generation)  
    - Random Undersampling (majority reduction)
    
    **3. Threshold Tuning**
    Default 0.5 threshold is arbitrary. Optimal threshold was found 
    using Precision-Recall curve maximizing F1 score.
    
    **4. Full Deployment Pipeline**
    - Flask REST API serving predictions
    - Streamlit interactive frontend
    - Deployable to Render/Railway
    
    ### Tech Stack
    `Python` `XGBoost` `scikit-learn` `imbalanced-learn` 
    `Flask` `Streamlit` `pandas` `numpy` `scipy`
    
    ### Dataset
    [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 
    by ULB Machine Learning Group — 284,807 transactions, 492 fraud cases.
    """)