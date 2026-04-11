import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import os
import random

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
API_URL = os.getenv("API_URL", "http://localhost:5000/predict")
STATS_URL = API_URL.replace("/predict", "/stats")

LEGITIMATE_SAMPLES = [
    {
        'Time': 406.0, 'Amount': 149.62, 'Class': 0,
        'V1': -1.3598071336738, 'V2': -0.0727811733098497,
        'V3': 2.53634673796914, 'V4': 1.37815522427443,
        'V5': -0.338320769942518, 'V6': 0.462387777762292,
        'V7': 0.239598554061257, 'V8': 0.0986979012610507,
        'V9': 0.363786969611213, 'V10': 0.0907941719789316,
        'V11': -0.551599533260813, 'V12': -0.617800855762348,
        'V13': -0.991389847235408, 'V14': -0.311169353699879,
        'V15': 1.46817697209427, 'V16': -0.470400525259478,
        'V17': 0.207971241929242, 'V18': 0.0257905801985591,
        'V19': 0.403992960255733, 'V20': 0.251412098239705,
        'V21': -0.018306777944153, 'V22': 0.277837575558899,
        'V23': -0.110473910188767, 'V24': 0.0669280749146731,
        'V25': 0.128539358273528, 'V26': -0.189114843888824,
        'V27': 0.133558376740387, 'V28': -0.0210530534538215
    },
    {
        'Time': 8736.0, 'Amount': 2.69, 'Class': 0,
        'V1': 1.19185711131486, 'V2': 0.26615071205963,
        'V3': 0.16648011335321, 'V4': 0.448154078460911,
        'V5': 0.0600176492822243, 'V6': -0.0823608088155687,
        'V7': -0.0788029833323113, 'V8': 0.0851016549148104,
        'V9': -0.255425128109186, 'V10': -0.166974414004614,
        'V11': 1.61272666105479, 'V12': 1.06523531137287,
        'V13': 0.48909501589608, 'V14': -0.143772296441519,
        'V15': 0.635558093258208, 'V16': 0.463917041022171,
        'V17': -0.114804663102346, 'V18': -0.183361270123994,
        'V19': -0.145783041325259, 'V20': -0.0690831352230203,
        'V21': -0.225775248033138, 'V22': -0.638671952771851,
        'V23': 0.101287798171617, 'V24': -0.339846475529127,
        'V25': 0.167170404418143, 'V26': 0.125894532368176,
        'V27': -0.00898309914322813, 'V28': 0.0147241691924927
    },
    {
        'Time': 52312.0, 'Amount': 376.26, 'Class': 0,
        'V1': -0.966271711572087, 'V2': -0.185226008082898,
        'V3': 1.79299333957872, 'V4': -0.863291275036453,
        'V5': -0.0103088796030823, 'V6': 1.24720316752486,
        'V7': 0.23760893977178, 'V8': 0.377435874652262,
        'V9': -1.38702406270197, 'V10': -0.0549519224713749,
        'V11': -0.226487263835401, 'V12': 0.178228225877303,
        'V13': 0.507756869957169, 'V14': -0.287923596352353,
        'V15': -0.631418117709045, 'V16': -1.05964725286799,
        'V17': -0.684092786345479, 'V18': 1.96577500349538,
        'V19': -1.2326219700892, 'V20': -0.208037781160366,
        'V21': -0.108300452035545, 'V22': 0.00527359678253453,
        'V23': -0.190320518742841, 'V24': -1.17557533186321,
        'V25': 0.647376034602038, 'V26': -0.221928844458407,
        'V27': 0.0627228487293033, 'V28': 0.0614576285006353
    },
    {
        'Time': 75148.0, 'Amount': 21.86, 'Class': 0,
        'V1': -0.425965884614751, 'V2': 0.960523044882985,
        'V3': 1.14110934559042, 'V4': -0.168252079753527,
        'V5': 0.420987592893098, 'V6': -0.0297061667735579,
        'V7': 0.476200949078359, 'V8': 0.0679009893278773,
        'V9': -0.338266835965518, 'V10': -0.117433994491897,
        'V11': -0.183542816762541, 'V12': -0.145786956492015,
        'V13': -0.0690831352230203, 'V14': -0.225775248033138,
        'V15': -0.638671952771851, 'V16': 0.101287798171617,
        'V17': -0.339846475529127, 'V18': 0.167170404418143,
        'V19': 0.125894532368176, 'V20': -0.00898309914322813,
        'V21': 0.0147241691924927, 'V22': 0.0614576285006353,
        'V23': -0.221928844458407, 'V24': 0.647376034602038,
        'V25': -1.17557533186321, 'V26': -0.190320518742841,
        'V27': 0.00527359678253453, 'V28': -0.108300452035545
    },
    {
        'Time': 125317.0, 'Amount': 44.80, 'Class': 0,
        'V1': 0.304455536355838, 'V2': 0.0849684725895812,
        'V3': 0.485514169249848, 'V4': 0.970480756586405,
        'V5': -0.152879073454155, 'V6': 0.562320439764783,
        'V7': 0.372842560492922, 'V8': -0.236357726766826,
        'V9': 0.241754936625696, 'V10': -0.0834985534684766,
        'V11': 0.551462028869843, 'V12': 0.345586266655985,
        'V13': -0.263667882949827, 'V14': 0.0642850847787982,
        'V15': 0.0189569835027791, 'V16': 0.249824451408225,
        'V17': -0.0428479680990894, 'V18': 0.406077994340154,
        'V19': 0.0440820277793626, 'V20': 0.14723394822643,
        'V21': 0.0965452867487008, 'V22': 0.0882042993829225,
        'V23': -0.12977607583991, 'V24': 0.134820633762565,
        'V25': 0.0386011099016805, 'V26': 0.0167280024107437,
        'V27': 0.0159906628946484, 'V28': 0.0165174948888563
    }
]

FRAUD_SAMPLES = [
    {
        'Time': 406.0, 'Amount': 1.00, 'Class': 1,
        'V1': -2.3122265423263, 'V2': 1.95199201064158,
        'V3': -1.60985073229769, 'V4': 3.9979055875468,
        'V5': -0.522187864667764, 'V6': -1.42654531920595,
        'V7': -2.53738730624579, 'V8': 1.39165724829804,
        'V9': -2.77008927719433, 'V10': -2.77227214465915,
        'V11': 3.20203320709635, 'V12': -2.89990738849473,
        'V13': -0.595221881324605, 'V14': -4.28925378244217,
        'V15': 0.389724120274487, 'V16': -1.14074717980657,
        'V17': -2.83005567450437, 'V18': -0.0168224681808257,
        'V19': 0.416955705037907, 'V20': 0.126910559061474,
        'V21': 0.517232370861764, 'V22': -0.0350493686052974,
        'V23': -0.465211076944875, 'V24': 0.320198197836216,
        'V25': 0.0445191674731724, 'V26': 0.177839798284401,
        'V27': 0.261145002567677, 'V28': -0.143275874698918
    },
    {
        'Time': 2.0, 'Amount': 378.66, 'Class': 1,
        'V1': -1.35980713367388, 'V2': -0.0727811733098497,
        'V3': 2.53634673796914, 'V4': 1.37815522427443,
        'V5': -0.338320769942518, 'V6': 0.462387777762292,
        'V7': 0.239598554061257, 'V8': 0.0986979012610507,
        'V9': -2.77008927719433, 'V10': -2.77227214465915,
        'V11': 3.20203320709635, 'V12': -2.89990738849473,
        'V13': -0.595221881324605, 'V14': -4.28925378244217,
        'V15': 0.389724120274487, 'V16': -1.14074717980657,
        'V17': -2.83005567450437, 'V18': -0.0168224681808257,
        'V19': 0.416955705037907, 'V20': 0.126910559061474,
        'V21': 0.517232370861764, 'V22': -0.0350493686052974,
        'V23': -0.465211076944875, 'V24': 0.320198197836216,
        'V25': 0.0445191674731724, 'V26': 0.177839798284401,
        'V27': 0.261145002567677, 'V28': -0.143275874698918
    },
    {
        'Time': 1.0, 'Amount': 26.40, 'Class': 1,
        'V1': -2.3122265423263, 'V2': 1.95199201064158,
        'V3': -1.60985073229769, 'V4': 3.9979055875468,
        'V5': -0.522187864667764, 'V6': -1.42654531920595,
        'V7': -2.53738730624579, 'V8': 1.39165724829804,
        'V9': -2.77008927719433, 'V10': -2.77227214465915,
        'V11': 3.20203320709635, 'V12': -2.89990738849473,
        'V13': -0.595221881324605, 'V14': -4.28925378244217,
        'V15': 0.389724120274487, 'V16': -1.14074717980657,
        'V17': -2.83005567450437, 'V18': -0.0168224681808257,
        'V19': 0.416955705037907, 'V20': 0.126910559061474,
        'V21': 0.517232370861764, 'V22': -0.0350493686052974,
        'V23': -0.465211076944875, 'V24': 0.320198197836216,
        'V25': 0.0445191674731724, 'V26': 0.177839798284401,
        'V27': 0.261145002567677, 'V28': -0.143275874698918
    },
    {
        'Time': 57.0, 'Amount': 529.00, 'Class': 1,
        'V1': -1.27887804885193, 'V2': -0.190320518742841,
        'V3': 0.647376034602038, 'V4': -1.17557533186321,
        'V5': -0.221928844458407, 'V6': -2.53738730624579,
        'V7': 1.39165724829804, 'V8': -0.338266835965518,
        'V9': -0.117433994491897, 'V10': -3.77227214465915,
        'V11': 2.20203320709635, 'V12': -3.89990738849473,
        'V13': -0.495221881324605, 'V14': -5.28925378244217,
        'V15': 0.289724120274487, 'V16': -2.14074717980657,
        'V17': -3.83005567450437, 'V18': -0.1168224681808257,
        'V19': 0.316955705037907, 'V20': 0.026910559061474,
        'V21': 0.417232370861764, 'V22': -0.1350493686052974,
        'V23': -0.365211076944875, 'V24': 0.220198197836216,
        'V25': 0.0345191674731724, 'V26': 0.077839798284401,
        'V27': 0.161145002567677, 'V28': -0.243275874698918
    },
    {
        'Time': 3572.0, 'Amount': 0.77, 'Class': 1,
        'V1': -2.3122265423263, 'V2': 1.95199201064158,
        'V3': -1.60985073229769, 'V4': 3.9979055875468,
        'V5': -0.622187864667764, 'V6': -1.52654531920595,
        'V7': -2.63738730624579, 'V8': 1.29165724829804,
        'V9': -2.87008927719433, 'V10': -2.87227214465915,
        'V11': 3.10203320709635, 'V12': -2.99990738849473,
        'V13': -0.695221881324605, 'V14': -4.38925378244217,
        'V15': 0.289724120274487, 'V16': -1.24074717980657,
        'V17': -2.93005567450437, 'V18': -0.1168224681808257,
        'V19': 0.316955705037907, 'V20': 0.026910559061474,
        'V21': 0.417232370861764, 'V22': -0.1350493686052974,
        'V23': -0.565211076944875, 'V24': 0.220198197836216,
        'V25': 0.0345191674731724, 'V26': 0.077839798284401,
        'V27': 0.161145002567677, 'V28': -0.143275874698918
    }
]


@st.cache_data
def load_test_data():
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'creditcard.csv'))
        return df
    except FileNotFoundError:
        return None


@st.cache_data
def load_comparison():
    try:
        comp = joblib.load(os.path.join(BASE_DIR, 'models', 'model_comparison.pkl'))
        return comp
    except:
        return None


df = load_test_data()
comparison_df = load_comparison()


st.title("🔍 Credit Card Fraud Detection System")
st.markdown("This system uses an **XGBoost model** trained on 284,000+ real transactions to detect fraudulent activity in real time.")

st.warning("""
⚠️ **Heads up:** This app runs on free-tier hosting. If the first prediction takes 30–60 seconds,
the backend server is waking up from sleep. Wait a moment and try again — subsequent predictions will be fast.
""")


st.sidebar.title("⚙️ Model Settings")

custom_threshold = st.sidebar.slider(
    "Detection Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Controls the tradeoff between catching fraud and generating false alarms"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Lower threshold →**
- Catches MORE fraud ✅
- More false alarms ❌

**Higher threshold →**
- Fewer false alarms ✅
- Misses MORE fraud ❌

This is the core business tradeoff in every real fraud system.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
st.sidebar.metric("Algorithm", "XGBoost")
st.sidebar.metric("ROC-AUC", "0.9788")
st.sidebar.metric("F1 Score", "0.8409")
st.sidebar.metric("Precision", "91.4%")

st.sidebar.markdown("---")
st.sidebar.markdown("### Live API Stats")
try:
    stats_resp = requests.get(STATS_URL, timeout=5).json()
    st.sidebar.metric("Total Predictions", stats_resp.get('total_predictions', 0))
    st.sidebar.metric("Fraud Detected", stats_resp.get('fraud_detected', 0))
    st.sidebar.metric("Legitimate", stats_resp.get('legitimate_detected', 0))
except:
    st.sidebar.caption("API stats unavailable — server may be sleeping")


tab1, tab2, tab3 = st.tabs([
    "🎲 Test with Real Data",
    "📊 Model Comparison",
    "ℹ️ About This Project"
])


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

            if df is not None:
                if sample_type == "Random FRAUD Transaction":
                    row = df[df['Class'] == 1].sample(1).iloc[0].to_dict()
                elif sample_type == "Random LEGITIMATE Transaction":
                    row = df[df['Class'] == 0].sample(1).iloc[0].to_dict()
                else:
                    row = df.sample(1).iloc[0].to_dict()
                sample = row
            else:
                if sample_type == "Random FRAUD Transaction":
                    sample = random.choice(FRAUD_SAMPLES)
                elif sample_type == "Random LEGITIMATE Transaction":
                    sample = random.choice(LEGITIMATE_SAMPLES)
                else:
                    sample = random.choice(FRAUD_SAMPLES + LEGITIMATE_SAMPLES)

            st.session_state['sample'] = sample
            st.session_state['true_label'] = int(sample['Class'])
            st.session_state['last_probability'] = None

        if 'sample' in st.session_state:
            true_label = st.session_state['true_label']
            if true_label == 1:
                st.error("**True Label: 🚨 FRAUD**")
            else:
                st.success("**True Label: ✅ LEGITIMATE**")
            st.caption("Transaction loaded. Click Analyze to get a prediction.")

    with col2:
        st.subheader("Transaction Details")
        if 'sample' in st.session_state:
            amount = st.session_state['sample']['Amount']
            time_val = st.session_state['sample']['Time']
            hour = (time_val / 3600) % 24
            st.metric("Amount", f"${amount:.2f}")
            st.metric("Time", f"{time_val:.0f} seconds")
            st.metric("Hour of Day", f"{hour:.1f}:00")
        else:
            st.info("Load a sample transaction to see details")

    st.markdown("---")

    if 'last_probability' in st.session_state and st.session_state['last_probability'] is not None:
        fraud_prob = st.session_state['last_probability']
        prediction = "fraud" if fraud_prob >= custom_threshold else "legitimate"

        st.subheader("🎯 Live Threshold Preview")
        st.markdown(f"Last analyzed probability was **{fraud_prob*100:.2f}%**. At your current threshold of **{custom_threshold}**, this transaction is classified as:")

        if prediction == "fraud":
            st.error(f"### 🚨 FRAUD — move threshold above {fraud_prob:.2f} to reclassify as legitimate")
        else:
            st.success(f"### ✅ LEGITIMATE — move threshold below {fraud_prob:.2f} to flag as fraud")

        display_prob = max(fraud_prob, 0.01)
        st.progress(display_prob)
        st.caption(f"Fraud probability: {fraud_prob*100:.2f}% | Current threshold: {custom_threshold}")
        st.markdown("---")

    if st.button("🔍 Analyze Transaction", use_container_width=True, type="primary"):

        if 'sample' not in st.session_state:
            st.warning("Please load a sample transaction first.")
        else:
            sample = st.session_state['sample']

            payload = {
                "Time": float(sample['Time']),
                "Amount": float(sample['Amount'])
            }
            for i in range(1, 29):
                payload[f'V{i}'] = float(sample[f'V{i}'])

            with st.spinner("Contacting prediction API..."):
                try:
                    response = requests.post(API_URL, json=payload, timeout=15)
                    result = response.json()

                    fraud_prob = result['fraud_probability']
                    st.session_state['last_probability'] = fraud_prob
                    prediction = "fraud" if fraud_prob >= custom_threshold else "legitimate"

                    st.subheader("🤖 Model Prediction")

                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        if prediction == "fraud":
                            st.error("### 🚨 FRAUD DETECTED")
                        else:
                            st.success("### ✅ LEGITIMATE")

                    with col_b:
                        st.metric("Fraud Probability", f"{fraud_prob*100:.2f}%")

                    with col_c:
                        st.metric("Risk Level", result['risk_level'])

                    display_prob = max(fraud_prob, 0.01)
                    st.progress(display_prob)
                    st.caption(f"Fraud probability bar — threshold set to {custom_threshold}")

                    if 'true_label' in st.session_state:
                        true = st.session_state['true_label']
                        pred_int = 1 if prediction == "fraud" else 0

                        st.markdown("---")
                        if true == pred_int:
                            st.success(f"✅ **CORRECT** — Model correctly identified this as {'fraud' if true == 1 else 'legitimate'}")
                        else:
                            if true == 1 and pred_int == 0:
                                st.error("❌ **FALSE NEGATIVE** — Actual fraud that the model missed. In a real system this means money lost.")
                            else:
                                st.warning("⚠️ **FALSE POSITIVE** — Legitimate transaction flagged as fraud. In a real system this means an unhappy customer.")

                    st.info(f"""
                    **Threshold Analysis:**
                    The model assigned a fraud probability of **{fraud_prob*100:.2f}%**.
                    At your current threshold of **{custom_threshold}**, this is **{prediction}**.
                    Move the slider in the sidebar to see how the decision changes without re-running the model.
                    """)

                    st.markdown("---")
                    st.markdown("**Raw API Response:**")
                    st.json(result)

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach the prediction API.")
                    st.info("""
                    **Running locally?** Start Flask first:
                    python app/api/app.py
                    **On the deployed app?** The backend is sleeping. Wait 30 seconds and try again.
                    """)

                except requests.exceptions.Timeout:
                    st.error("⏱️ The request timed out.")
                    st.info("The server is taking too long to respond. This usually means it just woke up from sleep. Wait 30 seconds and try again.")

                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")


with tab2:
    st.header("📊 Model Comparison")
    st.markdown("Five different approaches were trained and evaluated on the same test set. Here are the results.")

    if comparison_df is not None:
        display_cols = ['model', 'f1', 'roc_auc', 'precision',
                        'recall', 'true_positives', 'false_negatives',
                        'false_positives']
        available_cols = [c for c in display_cols if c in comparison_df.columns]
        st.dataframe(
            comparison_df[available_cols].style.highlight_max(
                subset=[c for c in ['f1', 'roc_auc'] if c in available_cols],
                color='lightgreen'
            ).highlight_min(
                subset=[c for c in ['false_negatives', 'false_positives'] if c in available_cols],
                color='lightgreen'
            ),
            use_container_width=True
        )
    else:
        results_data = {
            'Model': ['LR Baseline', 'LR Class Weights',
                      'RF + SMOTE', 'RF + Undersampling', 'XGBoost'],
            'F1': [0.7000, 0.1049, 0.3480, 0.1685, 0.8409],
            'ROC-AUC': [0.9561, 0.9636, 0.9726, 0.9694, 0.9788],
            'Precision': [0.8615, 0.0558, 0.2201, 0.0933, 0.9136],
            'Recall': [0.5895, 0.8737, 0.8316, 0.8737, 0.7789],
            'Fraud Caught': [56, 83, 79, 83, 74],
            'Fraud Missed': [39, 12, 16, 12, 21],
            'False Alarms': [9, 1404, 280, 807, 7]
        }
        st.dataframe(pd.DataFrame(results_data), use_container_width=True)

    st.markdown("---")
    st.subheader("How to Read These Results")

    col1, col2 = st.columns(2)

    with col1:
        st.error("""
        **The Accuracy Trap**

        Predicting ALL transactions as legitimate gives 99.83% accuracy
        while catching zero fraud. This is why accuracy is never used
        in fraud detection. F1, Precision, Recall, and ROC-AUC tell
        the real story.
        """)

    with col2:
        st.warning("""
        **The Core Tradeoff**

        LR Class Weights catches 83 frauds ✅ but triggers 1,404 false alarms ❌

        XGBoost catches 74 frauds with only 7 false alarms.

        The right choice depends on what the business values more —
        catching every fraud or protecting the customer experience.
        XGBoost was chosen for deployment because 91% precision
        makes it operationally realistic.
        """)

    st.markdown("---")
    st.subheader("Why These Techniques Were Compared")

    st.markdown("""
    **Class Weights** — the simplest fix. Tells the model to penalize
    mistakes on fraud cases more heavily. Costs nothing computationally.
    Always the first thing to try on an imbalanced dataset.

    **SMOTE** — creates new synthetic fraud transactions by interpolating
    between existing ones in feature space. Gives the model more fraud
    examples to learn from without just duplicating existing ones.

    **Undersampling** — removes legitimate transactions randomly until
    classes are balanced. Fast but wastes data — you're throwing away
    information that helps the model understand legitimate behavior.

    **XGBoost with scale_pos_weight** — handles imbalance natively.
    Treats each fraud example as if it were worth 568 legitimate examples
    during training. Combined with sequential error correction (each tree
    fixing the mistakes of the previous one), this gives the best overall result.
    """)


with tab3:
    st.header("ℹ️ About This Project")

    st.markdown("""
    ## The Problem

    Credit card fraud costs the global financial system billions every year.
    Traditional rule-based systems — "flag any transaction over $500 at 2am" —
    miss sophisticated attacks and generate too many false alarms. Machine
    learning approaches learn directly from historical fraud patterns and
    adapt to new attack methods automatically.

    ## The Dataset

    [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
    by the ULB Machine Learning Group. 284,807 transactions over two days,
    of which 492 are fraudulent — just 0.17%. This extreme imbalance is the
    central technical challenge of the project.

    ## What V1 Through V28 Actually Are

    The 28 features labeled V1–V28 are the result of a mathematical technique
    called PCA (Principal Component Analysis). The bank that provided this
    dataset applied PCA to the original transaction features — things like
    merchant category, location, card type, and purchase history — before
    releasing the data publicly.

    PCA transforms correlated features into a smaller set of uncorrelated
    components that capture the same underlying patterns. The bank did this
    specifically to protect cardholder privacy. The original feature names
    and what they represent cannot be recovered from V1–V28 alone. This
    is standard practice in financial ML when sharing data externally.

    Despite not knowing what V14 represents in the real world, the model
    discovered it is by far the strongest predictor of fraud — contributing
    40.7% of XGBoost's decision-making. The KS test in the EDA phase
    statistically confirmed this before any model was trained.

    ## Key Technical Decisions

    **RobustScaler over StandardScaler** — the Amount feature had a skewness
    of 17 and kurtosis above 90, meaning extreme outliers were common.
    StandardScaler uses mean and standard deviation, both badly distorted
    by outliers. RobustScaler uses median and interquartile range instead,
    making it resistant to those extremes.

    **Cyclical time encoding** — raw seconds since the first transaction
    is nearly useless as a feature. Converting to hour of day and then
    applying sine/cosine encoding means the model understands that 11pm
    and 1am are two hours apart, not 22 hours apart as raw numbers would suggest.

    **Threshold tuning** — the default classification threshold of 0.5 is
    arbitrary. The optimal threshold was found by evaluating F1 at every
    possible threshold using the Precision-Recall curve. This is what the
    slider in this app controls.

    **Fit scaler on training data only** — the RobustScaler was fitted
    exclusively on training data and then applied to test data without
    refitting. Fitting on test data would constitute data leakage and
    produce dishonestly optimistic evaluation metrics.

    ## Tech Stack

    `Python` `XGBoost` `scikit-learn` `imbalanced-learn` `scipy`
    `Flask` `Streamlit` `pandas` `numpy` `joblib` `gunicorn`
    `GitHub` `Render`
    """)